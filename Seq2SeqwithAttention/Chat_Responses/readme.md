#######################################################################################

The model is trained on GRU based Seq2Seq Architecture with Luong attention.

It is observed that the model response is not as good as expected as the model was trained after various pre processing steps. It was expected that the model can frame entire sentence as the architecture was integrated with Luong attention.

However the model output is far better than the LSTM based Seq2Seq architecture without attention.

#######################################################################################

Here are some sample conversation with the model where the model was asked various finance related questions.

---------------------------------------------------------------------------------------

Hello! Financeot is ready to solve your query! Type 'quit' to exit.
> Why do some stocks have a higher margin requirement?
Bot: some you can have a successful mobile of parts a test that can help you as much as possible . .
> why do I need an emergency fund if I already have investments?
Bot: i believe you believe and what you have that can have you been . it is always good to have have you to have . .
> How would bonds fare if interest rates rose?
Bot: there are three players you need to take a few steps . .
> Simple and safe way to manage a lot of cash
Bot: . what are the of you are ? .
> Personal Tax Return software for Linux?
Bot: the criteria for you for is as follows as of and and and and . and .
> Is it possible to make money by getting a mortgage?
Bot: it is not a common step to make it easy to pay . you are here to you pay about your money . you you you .
> Is it ever a good idea to close credit cards?
Bot: it is not a good step but you need to buy your budget but you can give it up . . you
> Do I not have a credit score?
Bot: i have not a few few but not not not . it does not have been very few . not not not . not .
> What can cause rent prices to fall?
Bot: . have a comprehensive type of reasons to the task or open a few situation . . . .
> What is a good asset allocation for a 25 year old?
Bot: . old to old old year old year . year year .
> Summarize the functions of the Federal Reserve.
Bot: the main of is a .
> Create a container class that holds two values.
Bot: class blue two two two two two two two ? two . a .
> Write a short poem describing a setting sun.
Bot: the sun sun sun
> Generate a comment on a science article.
Bot: the science of the science science the science . science . .
> quit

#######################################################################################

Again the research will get contineud towards higher accuracy so that model can generate human like response. The next plan is to train the LSTM based Seq2Seq Architeture.

#######################################################################################

All the model experiments are logged into wandb for future refernce and tracking.

#######################################################################################

Reference : The model architecture was highly inspired from pytorch official documentaion on chatbot training. All the refernces with sources are mentioned in the eritten report.

#######################################################################################