# Machine Learning Projet: Barbell lift
The goal of this project is to use data from various accelerometers to tell if the user performs barbell lifts correctly or incorrectly (5 different ways).
<br>
#### The results are:
  A - correct<br>
  B - throwing elbows to the front<br>
  C - lifting the dumbbell only halfway<br>
  D - lowering the dumbbell only halfway<br>
  E - throwing hips to the front<br>
##### Software and Libraries
* Python 3.5.2
* pandas
* scikit-learn
##### Introduction
The non-preprocessed dataset for this project is provided in the same repository with the name **pml-training.csv** The dataset consists of 19622 entries of the accelerometers in the x, y and z coordinates making a total of 153 such features. There are six other features making a total of 159 features and hence a table with **19622 rows and 160 columns**. 
##### Preprocessing steps
* There are `100` columns that have `19216` missing values each and hence need to be removed from the dataset.
* The columns `Unnamed: 0`, `raw_timestamp_part_1`, `cvtd_timestamp`, `user_name`, `raw_timestamp_part_2`, `new_window`, `num_window` do not convey any information that is helpful in predictions and hence are removed.
* `classe` is the `target/label/class` column and the string classes have been mapped to integers as follows:
  * A -> 0
  * B -> 1
  * C -> 2
  * D -> 3
  * E -> 4
##### Performance metric
The performance metric used for this project is the standard [*k-fold cross-validation*](https://en.wikipedia.org/wiki/Cross-validation_(statistics)#k-fold_cross-validation) method with k=10 & 5.<br>
##### Learning method
The classification method used is a `Random Forest` using [*gini*](https://en.wikipedia.org/wiki/Gini_coefficient) method with `min_impurty_split` or minimum gini level to initiate a split set to 0.1 and a `max_depth` or maximum permissible depth of the decision tree set to 15. The classifier in this project uses all the available CPUs.
##### Accuracy score
The *k-fold cross-validation* used in this project gives the mean accuracy score around **78.79%** for `k = 10` and **70.05%** for `k = 5`
