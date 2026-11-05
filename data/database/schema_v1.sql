-- Fictional AcmeOps schema for database analysis and schema comparison.
CREATE TABLE users (user_id INTEGER PRIMARY KEY, email VARCHAR(255) NOT NULL UNIQUE, role VARCHAR(40) NOT NULL);
CREATE TABLE workspaces (workspace_id INTEGER PRIMARY KEY, name VARCHAR(100) NOT NULL);
CREATE TABLE memberships (user_id INTEGER NOT NULL, workspace_id INTEGER NOT NULL, PRIMARY KEY(user_id, workspace_id), FOREIGN KEY(user_id) REFERENCES users(user_id), FOREIGN KEY(workspace_id) REFERENCES workspaces(workspace_id));
CREATE TABLE tickets (ticket_id INTEGER PRIMARY KEY, title VARCHAR(255) NOT NULL, status VARCHAR(32) NOT NULL, severity VARCHAR(20) NOT NULL, reported_by INTEGER NOT NULL, FOREIGN KEY(reported_by) REFERENCES users(user_id));
CREATE TABLE releases (release_id INTEGER PRIMARY KEY, version VARCHAR(20) NOT NULL UNIQUE, released_at DATE NOT NULL);
CREATE TABLE invoices (invoice_id INTEGER PRIMARY KEY, amount DECIMAL(12,2) NOT NULL, approval_count INTEGER NOT NULL DEFAULT 0, requested_by INTEGER NOT NULL, FOREIGN KEY(requested_by) REFERENCES users(user_id));
