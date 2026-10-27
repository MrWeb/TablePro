//
//  ObjectCopyTransactionTests.swift
//  TableProTests
//
//  The copy runs against a driver `withMetadataDriver` hands it, which is the connection's own
//  session driver on every engine that opts out of pooling (DuckDB, PGlite). A `BEGIN` of its own
//  over a transaction the user already had open aborts it on DuckDB and commits their pending work
//  on the engines that commit implicitly, so the copy joins the session's transaction instead, the
//  way `DataWriteExecutor`, `DatabaseManager+Principals` and `StructureRebuildPlanRunner` do.
//

import Foundation
@testable import TablePro
import TableProPluginKit
import Testing

private struct StatementFailure: Error {}

private final class CopyEventLog: @unchecked Sendable {
    private let lock = NSLock()
    private var entries: [String] = []

    var events: [String] { lock.withLock { entries } }

    func record(_ event: String) {
        lock.withLock { entries.append(event) }
    }
}

private final class RecordingCopyDriver: PluginDriverStub, @unchecked Sendable {
    private let name: String
    private let log: CopyEventLog
    private let sessionState: PluginSessionTransactionState
    private let failing: Set<String>
    private let failsInserts: Bool
    private let cancelsOnInsert: Bool

    init(
        _ name: String,
        log: CopyEventLog,
        sessionState: PluginSessionTransactionState = .idle,
        failing: Set<String> = [],
        failsInserts: Bool = false,
        cancelsOnInsert: Bool = false
    ) {
        self.name = name
        self.log = log
        self.sessionState = sessionState
        self.failing = failing
        self.failsInserts = failsInserts
        self.cancelsOnInsert = cancelsOnInsert
    }

    var capabilities: PluginCapabilities { [] }
    var supportsTransactions: Bool { true }
    var supportsTransactionalDDL: Bool { true }

    private func record(_ event: String) {
        log.record("\(name) \(event)")
    }

    func sessionTransactionState() async -> PluginSessionTransactionState { sessionState }

    func openSessionScope(for purpose: PluginSessionPurpose) async throws -> PluginSessionScope? {
        record("open \(purpose)")
        guard purpose == .relaxedForeignKeys else { return nil }
        return PluginSessionScope(purpose: purpose, saved: [], applied: [])
    }

    func closeSessionScope(_ scope: PluginSessionScope) async throws {
        record(Task.isCancelled ? "close \(scope.purpose) while cancelled" : "close \(scope.purpose)")
    }

    func execute(query: String) async throws -> PluginQueryResult {
        record(query)
        if failing.contains(query) { throw StatementFailure() }
        return PluginQueryResult(
            columns: ["id"], columnTypeNames: [], rows: [[.text("1")]], rowsAffected: 1, executionTime: 0
        )
    }

    func executeParameterized(query: String, parameters: [PluginCellValue]) async throws -> PluginQueryResult {
        record("insert")
        if cancelsOnInsert {
            withUnsafeCurrentTask { task in
                guard let task else { return }
                task.cancel()
            }
        }
        if failsInserts { throw StatementFailure() }
        return PluginQueryResult(columns: [], columnTypeNames: [], rows: [], rowsAffected: 1, executionTime: 0)
    }

    func beginTransaction() async throws { record("begin") }
    func commitTransaction() async throws { record("commit") }
    func rollbackTransaction() async throws { record("rollback") }
}

@MainActor
private final class StubMetadataProvider: ScopedMetadataProviding {
    private let drivers: [UUID: DatabaseDriver]

    init(drivers: [UUID: DatabaseDriver]) {
        self.drivers = drivers
    }

    func withMetadataDriver<T: Sendable>(
        scope: DatabaseScope,
        workload: MetadataConnectionPool.Workload,
        _ body: @Sendable @escaping (DatabaseDriver) async throws -> T
    ) async throws -> T {
        guard let driver = drivers[scope.connectionId] else {
            throw ObjectCopyError.refused("No driver for \(scope.connectionId)")
        }
        return try await body(driver)
    }

    func browseScope(for connectionId: UUID) -> DatabaseScope? { nil }
}

@MainActor
struct ObjectCopyTransactionTests {
    private let sourceScope = DatabaseScope(connectionId: UUID(), database: "app", schema: nil)
    private let targetScope = DatabaseScope(connectionId: UUID(), database: "staging", schema: nil)

    // Different engines, so the copier opens no scope of its own and the log holds only the runner's.
    private let sourceType = DatabaseType.postgresql
    private let targetType = DatabaseType.mysql

    private func endpoint(_ scope: DatabaseScope, _ type: DatabaseType) -> DatabaseEndpoint {
        DatabaseEndpoint(
            scope: scope,
            connectionName: scope.database,
            databaseType: type,
            safeModeLevel: .silent,
            color: .blue
        )
    }

    private func request(
        _ selections: [ObjectCopySelection],
        content: ObjectCopyContent,
        errorHandling: ImportErrorHandling = .stopAndRollback
    ) -> ObjectCopyRequest {
        ObjectCopyRequest(
            source: endpoint(sourceScope, sourceType),
            destination: .existing(endpoint(targetScope, targetType)),
            objects: selections,
            content: content,
            existingPolicy: .replace,
            errorHandling: errorHandling,
            wrapEachTableInTransaction: true
        )
    }

    private func dataStep(_ table: String, clears: Bool = false) -> ObjectCopyTableStep {
        ObjectCopyTableStep(
            selection: ObjectCopySelection(kind: .table, name: table, schema: nil),
            dropStatements: [],
            sequenceStatements: [],
            createStatements: [],
            truncateStatements: clears
                ? [SyncStatement(sql: "DELETE FROM \(table)", objectName: table, summary: "clear")]
                : [],
            columns: ["id"],
            primaryKeyColumns: ["id"],
            sourceQuery: "SELECT id FROM \(table)",
            targetTable: table,
            targetSchema: nil,
            estimatedRows: nil,
            copiesData: true,
            copiesIdentityColumn: false,
            note: nil
        )
    }

    private func dataPlan(
        _ steps: [ObjectCopyTableStep],
        errorHandling: ImportErrorHandling = .stopAndRollback
    ) -> ObjectCopyPlan {
        ObjectCopyPlan(
            request: request(steps.map(\.selection), content: .data, errorHandling: errorHandling),
            createsDatabase: false,
            tableSteps: steps,
            definitionSteps: []
        )
    }

    /// A replacement: the drop and the create are one unit, which is what makes the structure phase
    /// want a transaction of its own.
    private func replacementPlan(errorHandling: ImportErrorHandling = .stopAndRollback) -> ObjectCopyPlan {
        let selection = ObjectCopySelection(kind: .table, name: "orders", schema: nil)
        let step = ObjectCopyTableStep(
            selection: selection,
            dropStatements: [SyncStatement(sql: "DROP TABLE orders;", objectName: "orders", summary: "drop")],
            sequenceStatements: [],
            createStatements: [
                SyncStatement(sql: "CREATE TABLE orders (id INTEGER);", objectName: "orders", summary: "create"),
            ],
            truncateStatements: [],
            columns: [],
            primaryKeyColumns: ["id"],
            sourceQuery: "SELECT id FROM orders",
            targetTable: "orders",
            targetSchema: nil,
            estimatedRows: nil,
            copiesData: false,
            copiesIdentityColumn: false,
            note: nil
        )
        return ObjectCopyPlan(
            request: request([selection], content: .structure, errorHandling: errorHandling),
            createsDatabase: false,
            tableSteps: [step],
            definitionSteps: []
        )
    }

    private func run(
        _ plan: ObjectCopyPlan,
        source: RecordingCopyDriver,
        target: RecordingCopyDriver
    ) async throws -> ObjectCopyRunResult {
        let provider = StubMetadataProvider(drivers: [
            sourceScope.connectionId: PluginDriverAdapter(
                connection: TestFixtures.makeConnection(type: sourceType), pluginDriver: source
            ),
            targetScope.connectionId: PluginDriverAdapter(
                connection: TestFixtures.makeConnection(type: targetType), pluginDriver: target
            ),
        ])
        let runner = ObjectCopyRunner(manager: provider, gate: AlwaysAllowGate())
        return try await runner.run(plan, progress: ObjectCopyProgress(progress: Progress()))
    }

    // MARK: - Structure

    @Test("An idle session lets the copy open its own transaction")
    func idleSessionKeepsTheCopysOwnTransaction() async throws {
        let log = CopyEventLog()

        let result = try await run(
            replacementPlan(), source: RecordingCopyDriver("source", log: log), target: RecordingCopyDriver("target", log: log)
        )

        #expect(log.events == [
            "target begin", "target DROP TABLE orders;", "target CREATE TABLE orders (id INTEGER);", "target commit",
        ])
        #expect(result.pendingInSessionTransaction == false)
    }

    /// The statements still run: they join the transaction the user opened, and only the user can
    /// end it. Opening one over it is what aborted it on DuckDB.
    @Test("A session holding a transaction is joined rather than wrapped, with its keys as they are")
    func openSessionTransactionIsJoined() async throws {
        let log = CopyEventLog()
        let target = RecordingCopyDriver("target", log: log, sessionState: .inTransaction)

        let result = try await run(replacementPlan(), source: RecordingCopyDriver("source", log: log), target: target)

        #expect(log.events == ["target DROP TABLE orders;", "target CREATE TABLE orders (id INTEGER);"])
        #expect(result.pendingInSessionTransaction)
    }

    /// A `LOCK TABLES` holds no transaction to commit, but a `START TRANSACTION` would release it,
    /// so the copy sends none.
    @Test("A session holding locks is joined too")
    func sessionLocksAreJoined() async throws {
        let log = CopyEventLog()
        let target = RecordingCopyDriver("target", log: log, sessionState: .holdsSessionLocks)

        let result = try await run(replacementPlan(), source: RecordingCopyDriver("source", log: log), target: target)

        #expect(!log.events.contains("target begin"))
        #expect(result.pendingInSessionTransaction)
    }

    /// PostgreSQL commits an aborted transaction as a ROLLBACK, so a run that keeps going past a
    /// failure would report tables that were never created.
    @Test(
        "Only a run that promises a rollback puts the structure in a transaction",
        arguments: [ImportErrorHandling.skipAndContinue, .stopAndCommit]
    )
    func structureWithoutRollbackHasNoTransaction(errorHandling: ImportErrorHandling) async throws {
        let log = CopyEventLog()

        _ = try await run(
            replacementPlan(errorHandling: errorHandling),
            source: RecordingCopyDriver("source", log: log),
            target: RecordingCopyDriver("target", log: log)
        )

        #expect(log.events == [
            "target open relaxedForeignKeys",
            "target DROP TABLE orders;",
            "target CREATE TABLE orders (id INTEGER);",
            "target close relaxedForeignKeys",
        ])
    }

    // MARK: - Rows

    @Test("Rows are written with the target's keys relaxed inside its transaction, and the source left alone")
    func rowsAreRelaxedOnTheTargetInsideItsTransaction() async throws {
        let log = CopyEventLog()

        let result = try await run(
            dataPlan([dataStep("buildings")]),
            source: RecordingCopyDriver("source", log: log),
            target: RecordingCopyDriver("target", log: log)
        )

        #expect(result.rowsCopied == 1)
        #expect(log.events == [
            "target begin",
            "target open relaxedForeignKeys",
            "source SELECT id FROM buildings",
            "target insert",
            "target close relaxedForeignKeys",
            "target commit",
        ])
    }

    /// Relaxed, a DELETE would leave rows in a table outside the copy pointing at nothing.
    @Test("Clears run with the target's keys on, before the rows relax them")
    func clearsRunWithKeysOn() async throws {
        let log = CopyEventLog()

        _ = try await run(
            dataPlan([dataStep("buildings", clears: true)]),
            source: RecordingCopyDriver("source", log: log),
            target: RecordingCopyDriver("target", log: log)
        )

        #expect(log.events == [
            "target begin",
            "target DELETE FROM buildings",
            "target open relaxedForeignKeys",
            "source SELECT id FROM buildings",
            "target insert",
            "target close relaxedForeignKeys",
            "target commit",
        ])
    }

    @Test("A copy that throws puts the target's keys back before rolling back")
    func failedCopyRestoresKeys() async throws {
        let log = CopyEventLog()

        let result = try await run(
            dataPlan([dataStep("buildings")]),
            source: RecordingCopyDriver("source", log: log),
            target: RecordingCopyDriver("target", log: log, failsInserts: true)
        )

        #expect(result.failedCount == 1)
        #expect(log.events == [
            "target begin",
            "target open relaxedForeignKeys",
            "source SELECT id FROM buildings",
            "target insert",
            "target close relaxedForeignKeys",
            "target rollback",
        ])
    }

    @Test("A stopped copy puts the target's keys back outside the cancellation")
    func cancelledCopyRestoresKeys() async throws {
        let log = CopyEventLog()
        let plan = dataPlan([dataStep("buildings")])
        let source = RecordingCopyDriver("source", log: log)
        let target = RecordingCopyDriver("target", log: log, cancelsOnInsert: true)

        let result = try await Task { try await run(plan, source: source, target: target) }.value

        #expect(result.cancelled)
        #expect(log.events == [
            "target begin",
            "target open relaxedForeignKeys",
            "source SELECT id FROM buildings",
            "target insert",
            "target close relaxedForeignKeys",
            "target rollback",
        ])
    }

    @Test("Rows joining the session's own transaction keep its keys as they are")
    func sessionTransactionIsNotRelaxed() async throws {
        let log = CopyEventLog()

        let result = try await run(
            dataPlan([dataStep("buildings")]),
            source: RecordingCopyDriver("source", log: log),
            target: RecordingCopyDriver("target", log: log, sessionState: .inTransaction)
        )

        #expect(result.pendingInSessionTransaction)
        #expect(log.events == ["source SELECT id FROM buildings", "target insert"])
    }

    /// The table still holds its own rows, so the source's would be appended beside them. Past that
    /// failure the rows keep the target's keys, since a child's parent rows can be missing.
    @Test("Under Skip and continue a table whose clear failed gets no rows, and none are relaxed")
    func failedClearGetsNoRows() async throws {
        let log = CopyEventLog()
        let target = RecordingCopyDriver("target", log: log, failing: ["DELETE FROM buildings"])

        let result = try await run(
            dataPlan(
                [dataStep("estates", clears: true), dataStep("buildings", clears: true)],
                errorHandling: .skipAndContinue
            ),
            source: RecordingCopyDriver("source", log: log),
            target: target
        )

        #expect(result.failures.map { $0.outcome.selection.name } == ["buildings"])
        #expect(result.rowsCopied == 1)
        #expect(log.events == [
            "target DELETE FROM buildings",
            "target DELETE FROM estates",
            "source SELECT id FROM estates",
            "target insert",
        ])
    }
}
