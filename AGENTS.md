# Subagent Execution & Exit Protocol

1. **Scoped Responsibility:** If you are spawned as a subagent, you may ONLY modify files explicitly assigned to your task scope.
2. **Local Verification (Mandatory):** You are forbidden from marking a task complete until you have run the relevant build, linter, or test command in the terminal and confirmed a 0 exit code. Use the preconfigured virtual environment (.venv) for all Python commands.
3. **State Ledger Update:** Before terminating, you MUST update your assigned task row in `/STATE.md` with:
   - Status: `DONE` or `BLOCKED`
   - Modified Files: List of paths touched
   - Verification Command & Output Summary
4. **Hard Termination:** Once `/STATE.md` is updated, immediately return a concise 3-line summary to the orchestrator and terminate your turn. Do not ask follow-up questions or wait for acknowledgement.