# AGENTS.md - Development Agent Instructions

## Source of Truth

**PRIMARY REFERENCE:** SPEC Kit (https://github.com/github/spec-kit)
**SECONDARY REFERENCE:** SPEC_KIT_CONSTITUTION.md (local)

All development follows Spec-Driven Development methodology from SPEC Kit.
This document is INFERIOR to SPEC Kit - follow SPEC Kit unless SPEC_KIT_CONSTITUTION.md specifies otherwise.

## SPEC Kit Core Principles (MANDATORY)

- **DRY:** No code duplication - extract to shared modules
- **KISS:** Keep it simple stupid - prefer obvious over clever
- **CLEAN Architecture:** Domain → Application → Infrastructure → Presentation layers
- **GITFLOW:** feature/*, fix/*, release/*, hotfix/* branches from `develop`
- **TTD:** Test-Driven Development - L0 to L3, NO SKIPS allowed
- **Configuration:** ALL URLs from `.env` - NEVER hardcode

## Debugging & Screenshots

- **ALL debugging screenshots MUST be saved to:** `debugging/screenshots/` directory
- **NEVER commit screenshots to git** - they are already in `.gitignore`
- When taking screenshots during debugging, always use the `filename` parameter to save to `debugging/screenshots/`

## General Rules

- Always run lint/typecheck after making changes
- Never commit secrets or credentials
- Test that changes work before considering them complete
- NEVER skip tests - violates SPEC Kit constitution

## Service Credentials

### Grafana
- URL: http://localhost:3000
- Username: admin
- Password: admin123

## Test Execution (Per SPEC Kit)

```bash
# Run ALL tests (L0-L3) - NO SKIPS
pytest tests/ -v --tb=short

# Run specific level
pytest tests/test_l0_unit.py -v
pytest tests/test_l1_integration.py -v
pytest tests/test_l2_component.py -v
pytest tests/test_l3_e2e.py -v
```

## Pre-Commit Checklist

Before ANY commit, verify:
- [ ] All tests pass: `pytest tests/ -v --tb=short`
- [ ] No hardcoded URLs (grep for `https?://` in code)
- [ ] Configuration uses `os.getenv()`
- [ ] No secrets in code (.env in .gitignore)
- [ ] Code follows DRY, KISS, CLEAN principles
- [ ] Commit message follows Conventional Commits (feat:, fix:, refactor:, docs:, test:, chore:)