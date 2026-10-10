-- ============================================================
-- MasteryLoop — M1 schema
-- ============================================================
CREATE EXTENSION IF NOT EXISTS pgcrypto;
-- ============================================================
-- CURRICULUM
-- ============================================================
CREATE TABLE IF NOT EXISTS learners (
learner_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
external_ref VARCHAR(64) UNIQUE NOT NULL,
display_name VARCHAR(160) NOT NULL,
cohort VARCHAR(80),
enrolled_on DATE NOT NULL DEFAULT CURRENT_DATE,
is_active BOOLEAN NOT NULL DEFAULT TRUE
);
CREATE INDEX IF NOT EXISTS idx_learners_cohort ON learners(cohort);
CREATE TABLE IF NOT EXISTS skills (
skill_id VARCHAR(30) PRIMARY KEY, -- PY.COMP
skill_name VARCHAR(160) NOT NULL,
domain VARCHAR(80) NOT NULL,
depth INT NOT NULL DEFAULT 0, -- distance from a root skill
description TEXT
);
CREATE TABLE IF NOT EXISTS skill_prerequisites (
skill_id VARCHAR(30) NOT NULL REFERENCES skills(skill_id) ON DELETE CASCADE,
prerequisite_id VARCHAR(30) NOT NULL REFERENCES skills(skill_id) ON DELETE CASCADE,
strength NUMERIC(3,2) NOT NULL DEFAULT 1.0 CHECK (strength BETWEEN 0 AND 1),
PRIMARY KEY (skill_id, prerequisite_id),
CHECK (skill_id <> prerequisite_id) -- no self-loops; cycles checked by seed
);
-- ============================================================
-- ITEM BANK
-- ============================================================
CREATE TABLE IF NOT EXISTS items (
item_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
item_code VARCHAR(40) UNI
item_type VARCHAR(20) NOT NULL DEFAULT 'mcq'
CHECK (item_type IN ('mcq','multi_select','numeric','code','short_text')),
content_area VARCHAR(30) NOT NULL
CHECK (content_area IN ('fundamentals','application','analysis','synthesis')),
media_uri TEXT,
expected_seconds INT NOT NULL DEFAULT 60 CHECK (expected_seconds > 0),
status VARCHAR(20) NOT NULL DEFAULT 'draft'
CHECK (status IN ('draft','pilot','active','retired','flagged')),
authored_by VARCHAR(120),
created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_items_status_area ON items(status, content_area);
CREATE TABLE IF NOT EXISTS item_options (
option_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
item_id UUID NOT NULL REFERENCES items(item_id) ON DELETE CASCADE,
option_label CHAR(1) NOT NULL,
option_text TEXT NOT NULL,
is_correct BOOLEAN NOT NULL DEFAULT FALSE,
misconception VARCHAR(120), -- what choosing this distractor implies
UNIQUE (item_id, option_label)
);
-- exactly one correct option per single-answer item
CREATE UNIQUE INDEX IF NOT EXISTS idx_one_correct_option
ON item_options(item_id) WHERE is_correct;
CREATE TABLE IF NOT EXISTS item_skills (
item_id UUID NOT NULL REFERENCES items(item_id) ON DELETE CASCADE,
skill_id VARCHAR(30) NOT NULL REFERENCES skills(skill_id) ON DELETE CASCADE,
weight NUMERIC(3,2) NOT NULL DEFAULT 1.0 CHECK (weight BETWEEN 0 AND 1),
PRIMARY KEY (item_id, skill_id)
);
CREATE INDEX IF NOT EXISTS idx_item_skills_skill ON item_skills(skill_id);
CREATE TABLE IF NOT EXISTS item_calibrations (
calibration_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
item_id UUID NOT NULL REFERENCES items(item_id) ON DELETE CASCADE,
version INT NOT NULL CHECK (version >= 1),
discrimination NUMERIC(5,3) NOT NULL CHECK (discrimination > 0), -- a
difficulty NUMERIC(5,3) NOT NULL CHECK (difficulty BETWEEN -5 AND 5), -- b
guessing NUMERIC(4,3) NOT NULL DEFAULT 0 CHECK (guessing BETWEEN 0 AND 0.5), -- c
sample_size INT NOT NULL DEFAULT 0,
standard_error NUMERIC(5,3),
is_active BOOLEAN NOT NULL DEFAULT FALSE,
calibrated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
calibrator_version VARCHAR(40) NOT NULL DEFAULT 'm1-seed',
UNIQUE (item_id, version)
);
-- an item may have at most one active calibration at a time
CREATE UNIQUE INDEX IF NOT EXISTS idx_one_active_calibration
ON item_calibrations(item_id) WHERE is_active;
-- ============================================================
-- SESSIONS
-- ============================================================
CREATE TABLE IF NOT EXISTS assessments (
assessment_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
assessment_code VARCHAR(40) UNIQUE NOT NULL,
title VARCHAR(200) NOT NULL,
domain VARCHAR(80) NOT NULL,
min_items INT NOT NULL DEFAULT 8 CHECK (min_items >= 1),
max_items INT NOT NULL DEFAULT 30,
target_se NUMERIC(4,3) NOT NULL DEFAULT 0.300 CHECK (target_se > 0),
content_mix JSONB NOT NULL DEFAULT '{}',
is_active BOOLEAN NOT NULL DEFAULT TRUE,
CHECK (max_items >= min_items)
);
CREATE TABLE IF NOT EXISTS assessment_sessions (
session_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
assessment_id UUID NOT NULL REFERENCES assessments(assessment_id),
learner_id UUID NOT NULL REFERENCES learners(learner_id) ON DELETE CASCADE,
started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
completed_at TIMESTAMPTZ,
items_administered INT NOT NULL DEFAULT 0,
initial_theta NUMERIC(6,3) NOT NULL DEFAULT 0,
final_theta NUMERIC(6,3),
final_se NUMERIC(5,3),
stop_reason VARCHAR(30)
CHECK (stop_reason IN ('target_se_reached','max_items','abandoned',
'bank_exhausted','content_unsatisfiable')),
engine_version VARCHAR(40) NOT NULL DEFAULT 'm1-placeholder'
);
CREATE INDEX IF NOT EXISTS idx_sessions_learner ON assessment_sessions(learner_id, started_at DESC);
CREATE TABLE IF NOT EXISTS responses (
response_id BIGSERIAL PRIMARY KEY,
session_id UUID NOT NULL REFERENCES assessment_sessions(session_id) ON DELETE CASCADE,
item_id UUID NOT NULL REFERENCES items(item_id),
calibration_id UUID REFERENCES item_calibrations(calibration_id),
sequence_no INT NOT NULL CHECK (sequence_no >= 1),
selected_option CHAR(1),
response_text TEXT,
is_correct BOOLEAN NOT NULL,
theta_at_admin NUMERIC(6,3) NOT NULL,
latency_ms INT CHECK (latency_ms >= 0),
answered_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
UNIQUE (session_id, sequence_no),
UNIQUE (session_id, item_id) -- an item is never repeated inside a session
);
CREATE INDEX IF NOT EXISTS idx_responses_item ON responses(item_id, answered_at DESC);
CREATE TABLE IF NOT EXISTS ability_estimates (
id BIGSERIAL PRIMARY KEY,
session_id UUID NOT NULL REFERENCES assessment_sessions(session_id) ON DELETE CASCADE,
after_sequence INT NOT NULL CHECK (after_sequence >= 0),
theta NUMERIC(6,3) NOT NULL,
standard_error NUMERIC(5,3) NOT NULL CHECK (standard_error > 0),
estimator VARCHAR(20) NOT NULL DEFAULT 'mle',
estimated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
UNIQUE (session_id, after_sequence)
);
CREATE TABLE IF NOT EXISTS item_exposure (
item_id UUID NOT NULL REFERENCES items(item_id) ON DELETE CASCADE,
window_start DATE NOT NULL,
administered INT NOT NULL DEFAULT 0 CHECK (administered >= 0),
correct_count INT NOT NULL DEFAULT 0,
p_value NUMERIC(5,4), -- classical difficulty, for bank health review
PRIMARY KEY (item_id, window_start)
);
-- ============================================================
-- OUTCOMES
-- ============================================================
CREATE TABLE IF NOT EXISTS mastery_states (
learner_id UUID NOT NULL REFERENCES learners(learner_id) ON DELETE CASCADE,
skill_id VARCHAR(30) NOT NULL REFERENCES skills(skill_id) ON DELETE CASCADE,
skill_theta NUMERIC(6,3) NOT NULL,
standard_error NUMERIC(5,3) NOT NULL,
evidence_items INT NOT NULL DEFAULT 0,
status VARCHAR(20) NOT NULL DEFAULT 'unknown'
CHECK (status IN ('unknown','developing','proficient','mastered','at_risk')),
updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
PRIMARY KEY (learner_id, skill_id)
);
CREATE INDEX IF NOT EXISTS idx_mastery_status ON mastery_states(status);
CREATE TABLE IF NOT EXISTS recommendations (
recommendation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
learner_id UUID NOT NULL REFERENCES learners(learner_id) ON DELETE CASCADE,
session_id UUID REFERENCES assessment_sessions(session_id) ON DELETE SET NULL,
skill_id VARCHAR(30) NOT NULL REFERENCES skills(skill_id),
rank INT NOT NULL CHECK (rank >= 1),
rationale TEXT NOT NULL,
resource_uri TEXT,
issued_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
UNIQUE (learner_id, session_id, rank)
);