# Kcompetition

[WSDM KKBoxs - Churn Kaggle Competition](https://www.kaggle.com/c/kkbox-churn-prediction-challenge)

Competition deadline: **December, 17th, 2017**

## Files

- DataWrangling.ipynb: descriptive analytics
- data_manipulation.py: functions to clean and aggregate the data
- Churn Case Documentation.ipynb: document with explanation of our approach to the competition


## Useful Links

- [Reading and processing large .csv files](https://stackoverflow.com/questions/17444679/reading-a-huge-csv-file)
- [Using python module Dask for large .csv files](http://pythondata.com/dask-large-csv-python/)
    > [Dask documentation](https://dask.pydata.org/en/latest/)
- [KKBOx website](https://www.kkbox.com)
- [Great Pandas cheat sheet](https://github.com/pandas-dev/pandas/blob/master/doc/cheatsheet/Pandas_Cheat_Sheet.pdf)
- [Numpy cheatsheet](https://www.dataquest.io/blog/numpy-cheat-sheet/)
- [Split big files into multiple files in Unix](https://eikhart.com/blog/autosplit-csv)

## Notes

(1) Remember to vectorize and use broadcasting techniques with numpy as much as possible
(2) Rearrange python notebook flow to start using files with new id from the beginning

## Files Locations and Folder Structure (Starting Point):

 ../
- /members.csv
- /sample_submission_zero.csv
- /test.csv
- /train.csv
- /transactions.csv
- **/KCompetition**
- > /KCompetition/DataWrangling.ipynb
- > /KCompetition/README.md
- > /KCompetition/transaction.py
- > /KCompetition/data_manipulation.py
- **/user_log_files**
- > /user_log_files/user_logs.csv

## Potential Game Plan For Submission

- Now till November 12th (1.5 weeks): Creating and fine tuning features
    - Lucas: user_logs file
    - Cathy: transactions file
- November 13th to 18th (1 week): Train model
- **November 18th: First Submission**
- November 19th to November 23rd (5 days): Rejigging model
- **November 23rd: Second Submission**
- November 24th to December 3rd (1 week): Rejigging model
    /November 24th to December 3rd (1 week): Lucas on vacation 😎/
- **December 3rd: Third Submission** (hopefully with new test data)
- December 4th to 17th (2 weeks): rejig and resubmit model as necessary
- **December 17th: Final Submission**
