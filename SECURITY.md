# Security Policy

## Supported Versions

This RAG application is actively maintained. Security updates are provided for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 1.x     | :white_check_mark: |
| < 1.0   | :x:                |

## Security Best Practices

### API Key Management
- **Never commit** your `.env` file to version control
- Store your `GOOGLE_API_KEY` securely in environment variables
- Rotate API keys regularly
- Use separate API keys for development and production
- Monitor API usage for unusual patterns

### Data Privacy
- PDFs are processed locally in memory and not stored permanently
- Chat history is stored only in session state (browser memory)
- No user data is transmitted to external services except Google's Gemini API
- Clear session data regularly using the "Clear All" button

### Deployment Security
- Use HTTPS in production deployments
- Set appropriate CORS policies
- Implement rate limiting to prevent abuse
- Monitor application logs for suspicious activity
- Keep dependencies updated regularly

### File Upload Security
- Maximum file size is enforced (50MB per file)
- Only PDF files are accepted
- File validation is performed before processing
- Malformed PDFs are handled gracefully with error messages

## Reporting a Vulnerability

If you discover a security vulnerability in this application, please report it responsibly:

1. **Do NOT** open a public GitHub issue
2. Email the maintainer directly at: [your-email@example.com]
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

### Response Timeline
- **Initial Response**: Within 48 hours
- **Status Update**: Within 7 days
- **Fix Timeline**: Depends on severity
  - Critical: Within 24-48 hours
  - High: Within 1 week
  - Medium: Within 2 weeks
  - Low: Next release cycle

### Disclosure Policy
- We follow responsible disclosure practices
- Security fixes will be released before public disclosure
- Credit will be given to reporters (unless anonymity is requested)

## Security Updates

Security updates are released as patch versions (e.g., 1.0.1, 1.0.2). Always use the latest version for the best security posture.

## Dependencies

This application uses third-party libraries. We regularly update dependencies to patch known vulnerabilities. Run `pip install -r requirements.txt --upgrade` to get the latest secure versions.

## Contact

For security concerns, contact: [your-email@example.com]
For general issues, use GitHub Issues: https://github.com/BlueZyne/rag_model/issues
