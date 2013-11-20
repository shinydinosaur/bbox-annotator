remember to be in the virtual env:
workon NAME_OF_ENV

To reset the database, run:
$ psql -U picasso -d picasso_annotations -f reset.sql
and authenticate as user 'picasso'
