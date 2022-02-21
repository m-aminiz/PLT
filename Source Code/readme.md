##### In The Name Of Allah


## Required Commands


### Virtual Enviornment

>pip install virtualenv

>virtualenv "name"

>"name"\Scripts\activate

>deactivate

>"name"\Scripts\activate.ps1

>Set-ExecutionPolicy Unrestricted -Force


### Freeze

>pip install freeze

>pip freeze > "filename(eg. requirements).txt"

>pip install -r "filename(eg. requirements).txt"


### Git

>git help

>git help "command_name"


>git init

>git status

>git log

>git diff HEAD

>git diff --staged


>git add "filename"

>git reset "filename"

>git add -A


>git commit

>git commit -m "massage"


>git branch

>git branch "branchname"

>git branch -d "branchname"

>git checkout "branchname"

>git merge "branchname"


>git remote (-v)

>git clone "repository_address"

>git remote add "remote_name" "remote_address"

>git push/pull (-u) "remote_name" "branch_name"


>git checkout -- "filename"

>git rm "filename"

>git show "commit_hash"


>git config (--global) user.name "user name"

>git config (--global) user.email "user email"

>git config (--global) core.editor "editor name"


>git tag

>git tag -a "tag_name" -m "tag message" (commit_hash)

>git show "tag_name"

>git push origin "tag_name"

>git push origin --tags

>git checkout "tag_name"



### Mongodb

>mongoimport -d "dbname" -c "collectionname" --file "filename".json

>mongoexport -d "dbname" -c "collectionname" --out "filename".json

>mongodump -d "databasename" -o "targetdirectory"

>mongorestore "targetdirectory" --db "databasename"



### Create Applicaton

>python setup.py bdist_msi

>pyinstaller --noconfirm --onefile --windowed PltApp.py

>python setup.py py2app

>shift = ds.BitsAllocated - j2k_precision