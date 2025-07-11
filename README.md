# COD-TASK1-DECISION-TREE-IMPLEMENTATION

*COMPANY*:COD TECH IT SOLUTIONS

*NAME*:VAIDHEESVARI.M.K

*INTERN ID*:CT06DG2683

*DOMAIN*:MACHINE LEARNING

*DURATION*:8 WEEKS

*MENTOR*:NEELA SANTHOSH

*This project focuses on implementing a machine learning model to classify iris flowers into three distinct species using the Decision Tree algorithm. The entire project is carried out in Python using essential libraries such as pandas, numpy, matplotlib, seaborn, and scikit-learn. A key objective of the project is not just to build a predictive model, but also to visualize the internal structure of the decision tree to understand how decisions are made.

The dataset used is the famous Iris dataset, one of the oldest and most well-known datasets in the field of machine learning. It was introduced by statistician Ronald A. Fisher in 1936 and has been widely used for demonstrating classification algorithms. The dataset contains a total of 150 samples, with each sample representing an iris flower. There are three species in the dataset: Iris setosa, Iris versicolor, and Iris virginica, with 50 samples for each class. Each flower is described using four numerical features: sepal length, sepal width, petal length, and petal width (all in centimeters).

The project begins by importing the necessary libraries and loading the Iris dataset using the load_iris() function provided by scikit-learn. The dataset is converted into a pandas DataFrame for easier data manipulation and visualization. An initial exploration of the dataset is conducted to examine the structure and contents, followed by the creation of a target_name column to display class names instead of just numerical labels.

Next, the dataset is divided into features (X) and target (y). The data is then split into training and testing subsets using the train_test_split() function, reserving 30% of the data for testing. This ensures that the model can be evaluated on unseen data.

A DecisionTreeClassifier from scikit-learn is initialized with a maximum depth of 3 to prevent overfitting and to keep the model simple and interpretable. The model is trained on the training data using the .fit() method. After training, predictions are made on the test data using the .predict() method.

To evaluate the model, metrics such as accuracy score and classification report are used. The classification report provides detailed performance information for each class, including precision, recall, and F1-score. The model typically achieves high accuracy on this dataset, usually above 95%, demonstrating that a decision tree is an effective classifier for this problem.

Finally, the structure of the trained decision tree is visualized using plot_tree() from sklearn.tree. This visual representation clearly shows how the model splits the data based on different feature values and how it arrives at its decisions.

In conclusion, this project demonstrates the effective use of the decision tree algorithm for multi-class classification. The Iris dataset provides an excellent foundation for learning classification techniques, data preprocessing, model training, evaluation, and visualization. This project is not only useful for academic learning but also showcases the importance of explainable AI models in real-world applications.*
