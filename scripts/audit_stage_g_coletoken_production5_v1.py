"""Run unchanged full D042 source/token/factual-provenance checks against the continuation ledger."""
import stage_g_coletoken_production_v1 as ledger
import audit_stage_g_production_d042 as audit
import run_stage_g_production_d042 as original_runner


def main():
    ledger.verify_manifest()
    audit.FOLDER = ledger.RUN
    audit.OUT = ledger.EXEC
    audit.check_inputs = ledger.checked_inputs
    audit.location = ledger.location
    audit.save_completed_business = ledger.save_completed_business
    original_runner.update_docs = ledger.update_docs
    report = audit.run(smoke=False)
    state = ledger.read(ledger.EXEC / 'execution_progress.json')
    pointer = ledger.read(ledger.POINTER)
    pointer.update(status=state['status'], completed_utc=ledger.now(),
                   completion_report=(ledger.RUN/'completion_report.json').relative_to(ledger.ROOT).as_posix())
    ledger.atomic(ledger.POINTER,pointer)
    return report


if __name__ == '__main__': main()
