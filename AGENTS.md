# AGENTS.md - Development Agent Instructions

## Debugging & Screenshots

- **ALL debugging screenshots MUST be saved to:** `debugging/screenshots/` directory
- **NEVER commit screenshots to git** - they are already in `.gitignore`
- When taking screenshots during debugging, always use the `filename` parameter to save to `debugging/screenshots/`

## General Rules

- Always run lint/typecheck after making changes
- Never commit secrets or credentials
- Test that changes work before considering them complete

## Service Credentials

### Grafana
- URL: http://localhost:3000
- Username: admin
- Password: admin123
