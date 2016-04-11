

library(ggplot2)
data("iris")
testing <- iris[c(14, 15, 17, 39, 41, 42, 2, 5, 9, 13, 38, 46, 50, 8, 11, 20, 32, 40, 27, 30, 31, 47, 6, 25, 45, 94, 65, 82, 60, 70, 54, 89, 95, 97, 75, 66, 76, 88, 52,56,67,69,79,86,107,55,59,87,77,127,53,122,78,102,142,143,146,148,116,140,149,113,117,138,129,133,135,137,121,105,103,101,110,106,123),  ]
testing2 <- testing
testing2$Row.Num <- rownames(testing2)
rf <- ifelse(testing2$Row.Num == 53 | testing2$Row.Num == 77 | testing2$Row.Num == 78 | testing2$Row.Num == 107, 0,1)
gbm <- ifelse(testing2$Row.Num == 127 | testing2$Row.Num == 107 | testing2$Row.Num == 78 | testing2$Row.Num == 53, 0,1)
nb <- ifelse(testing2$Row.Num == 107 | testing2$Row.Num == 78 | testing2$Row.Num == 53, 0,1)
nnet <- ifelse(testing2$Row.Num == 69 | testing2$Row.Num == 78 | testing2$Row.Num == 88, 0,1)
multinom <- ifelse(testing2$Row.Num == 69, 0,1)
svm <- ifelse(testing2$Row.Num == 42 | testing2$Row.Num == 107 | testing2$Row.Num == 135 | testing2$Row.Num == 69 | testing2$Row.Num == 78, 0,1)
library(shiny)
shinyServer(
  function(input, output) {
    output$plot1 <- renderPlot({
      data <- switch(input$varName, 
                     "Random Forest" = rf,
                     "Stochastic Gradient Boosting" = gbm,
                     "Naive Bayes" = nb,
                     "Neural Network" = nnet,
                     "Penalized Multinomial Regression" = multinom,
                     "Support Vector Machine" = svm)
      testing2$Misclass.Err <- data
      ggplot(testing2) + geom_point(aes(Petal.Length, Sepal.Length, colour = Misclass.Err, shape = Species), size = 2.5) + labs(x="Petal.Length", y = "Sepal.Length")
    })
    
    output$info <- renderPrint({
      # With ggplot2, no need to tell it what the x and y variables are.
      # threshold: set max distance, in pixels
      # maxpoints: maximum number of rows to return
      # addDist: add column with distance, in pixels
      nearPoints(testing2[, c(1:5)], input$plot_click, threshold = 10, maxpoints = 1,
                 addDist = TRUE)
    })
  }
)

