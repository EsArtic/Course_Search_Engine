The program is implement by python 3.6.
Before running the program, please make sure that there is no version conflict.

If you are using Linux/Mac OS, please use run.sh or save.sh to run the program.
If you are using Windows OS, please use run.bat or save.bat to run the program.

After running run.bat/run.sh, the output message will be display on the screen directly.
After running save.bat/save.sh, out output message will be redirect and write to the file "./output/vsm.out"

If you want to run the program directly, please note that it takes two required command line arguments,
and the usage is as following:

usage: Main.py [-h] -c COLLECTION -q QUERY

optional arguments:
  -h, --help            show this help message and exit
  -c COLLECTION, --collection COLLECTION
                        Path of the documents collection file
  -q QUERY, --query QUERY
                        Path of the queries collection file