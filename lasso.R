library(glmnet)

X <- read.csv("X.csv")
y <- read.csv("y.csv")



data(QuickStartExample)
x <- QuickStartExample$x
y <- QuickStartExample$y

res <- glmnet(x,y)
