-- scripts/roles.sql
-- Run this once as a superuser (postgres) to create a least-privilege app role.

DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'pqc_app') THEN
    CREATE ROLE pqc_app LOGIN PASSWORD 'change-me';
  END IF;
END $$;

-- Restrict privileges: only what the app needs.
GRANT CONNECT ON DATABASE pqcdb TO pqc_app;
GRANT USAGE ON SCHEMA public TO pqc_app;

GRANT SELECT, INSERT, UPDATE ON TABLE encrypted_records TO pqc_app;
GRANT INSERT ON TABLE audit_events TO pqc_app;

-- For BIGSERIAL sequences:
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO pqc_app;

-- Optional: if you add more tables later, you can grant on future tables:
-- ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT, INSERT, UPDATE ON TABLES TO pqc_app;
-- ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT USAGE, SELECT ON SEQUENCES TO pqc_app;
