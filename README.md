
### Linear Classification Using a SVM and a Decision tree

Perform classification on a dataset of social network ads,
determine whether a person will purchase an item through an ad shown on a social media app.

Pruning is then applied to ensure no overfitting for the decision tree.

###### Requirements:

    pip install -r requirements.txt
    
###### Usage:
    python ./src/SVM.py
    python ./src/DecisionTree.py

###### Example of a decision tree

![Example of a decision tree](/src/Decision_Tree.png)

###### Example of a pruned decision tree

![Example of a pruned decision tree](/src/Pruned_Decision_Tree.png)


To display a decision tree using *DecisionTree.py* ensure Graphviz is installed on you're system
See:

    https://stackoverflow.com/questions/35064304/runtimeerror-make-sure-the-graphviz-executables-are-on-your-systems-path-aft

Download:

    https://graphviz.org/download/