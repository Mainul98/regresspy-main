# Regresspy

Name: Md. Mainul Hasan

ID: 181-35-2339

`regresspy` is a python module for carrying out simple regressions using gradient descent algorithm. However, the library is half complete. Your task is to fill in the necessary codes. The tasks of the project is given below:
At first I have Fork this repository in the regresspy main file. Then I had had cloned this repository in my desktop from the pycharm terminal. Then I have changed the directory of regresspy. Then create a new branch of this repository Git add . using this command and all the file added in this branch on the other hand I left a comment and finally git push of this branch. Now I have updated the project I took first of all loss.py file and created some function as like mse, sse, mae and rmse and finally run this file from pycharm terminal. Then I go to my gradient descent file and change the forward function and run this function from pycharm terminal. Then I changed the test_loss.py file and make some functions (test_mase, test_rmse, test_sse) here my course teacher already created test_mae function so it easy for me to create another 3 function and finally i check those function from terminal and using same values. Then I go to regression.py file. 1-First of all i set weight and bias in predict function and return prediction.

2-Then get score all the (mse, rmse, mae, sse) function using if, else from the score function and return the score.

3-Actually i change self weight and bios but in the initialize_weights function but my course teacher already did it.

4-And finally i i change the train function using forward and backward function from the gradient descent file.

5-Using forward propagation (forward function) i got loss and print it.

Then finally i go to model.py file. 1-At first i changed the reshape in Y predict portion. I got error and again i set the reshape and run successfully.

2-I Used the stochastic gradient descent regressor and set iteration 20.

3-I trained X, Y through fit function from regression file of the stochastic gradient descent.

4-I predict stochastic gradient descent through predict function from regression.py file.

5-I calculated stochastic gradient descent rmse through rmse function from loss.py file.

6-And finally i got stochastic gradient descent rmse value, where the value is 0.5878859664089165

7-I evaluated regression value through regression function from regression.py file and set epoch/iteration 50 and also learning rate was 0.0001.

8-I trained X, Y through fit function from regression file of the regression value.

9-I predict regression value through predict function from regression.py file.

10-I calculated score of regression rmse through score function from regression.py file.

11-And finally i got RMSE value, where the value is 1.5323455704352724

Then i created the requirement.txt file that which the libraries actually needed of this project. And updated the requirement file. And i run this setup.py file on my cmd for the test. Finally add all the files in my branch on github using (git add .), left comment was "All the files "updated" using (git commit - m "") this command and push all files.