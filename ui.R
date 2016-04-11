library(shiny)

shinyUI(fluidPage(
  title = "Misclassified Data Points",
  plotOutput("plot1", click = "plot_click"),
  verbatimTextOutput("info"),
  
  fluidRow(
    column(width = 12,
           p(span("To gain an insight on the performance of the 
                  different algorithms available in the caret package, 
                  we compared the Misclassified data points of different
                  classifiers as it predicted on a testing set (50%) of 
                  the famous (Fisher's or Anderson's) iris data set. The
                  datapoints that were misclassified are revealed by a 
                  darker shade of", style = "color:red"), span("blue.", 
                                                               style = "color:dark blue"), span("The species to which
                                                                                                the datapoint belongs is designated by its shape. 
                                                                                                Identifying the datapoint visually can be challenging, 
                                                                                                but by using this App and clicking on the datapoint 
                                                                                                with the crosshair, it becomes easy. The full article 
                                                                                                can be found at ", style = "color:red"), tags$a(href=
                                                                                                                                                  "http://www.rpubs.com/DocOfi/170079", "Comparing Algorithms"),
             span("A companion guide to this App can be found at", style=
                    "color:red"), tags$a(href="http:www.rpubs.com/DocOfi/170080",
                                         "Comparing Algorithm - Guide to the App")
           )),
    
    fluidRow(
      column(width = 4, offset = 2,
             selectInput("varName", label = h3("Choose algorithm below"), 
                         choices = list("Random Forest", "Stochastic Gradient Boosting",
                                        "Naive Bayes", "Neural Network", "Penalized Multinomial Regression", "Support Vector Machine"), selected = "Random Forest"))
    )
    
    )))
