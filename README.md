# BLP-take-2-
This is another BLP code with fake DGP. The contraction mapping uses SQUAREM method to speed up the contraction in the inner loop. 
Here's the result:
'''
---Linear parameters from logit-IV regresion:---
             Coef.
Constant -3.643708
Price    -0.998815
Char1     5.726399
Char2     2.079959
---Start searching for random coefficients---
      Iter        f(X)
         1    0.397409
         2    0.293688
         3    0.293225
         4    0.292507
         5    0.292218
         6    0.292218
         7    0.292218
Optimization terminated successfully.
         Current function value: 0.292218
         Iterations: 7
         Function evaluations: 40
         Gradient evaluations: 10
             Coef.      S.E.
constant -3.485131  0.025390
price    -1.000877  0.004426
char1     6.000150  0.046532
char2     2.076059  0.014184
rc_char1  1.437001  0.153163
rc_char2  0.742560  0.200833
--- This estimation used 200 Halton draws. ---
--- 105.18553447723389 seconds ---
'''
