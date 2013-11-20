DELETE FROM annotations;
DELETE FROM tasks;
DELETE FROM users;

ALTER SEQUENCE annotations_id_seq RESTART;
ALTER SEQUENCE tasks_id_seq RESTART;
ALTER SEQUENCE tasks_id_seq RESTART;