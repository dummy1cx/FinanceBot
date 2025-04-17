The model architecture is based on the sequence-to-sequence (Seq2Seq) 
framework introduced by Sutskever et al. (2014), and implemented using 
PyTorch following its official tutorials. An LSTM-based encoder processes 
the input sequence and a decoder generates the output tokens step-by-step. 
Teacher forcing is applied to improve training efficiency.

Sample Model Output :
---------------------------------------------------------------------------------------
User: whaat is loan?
FinanceBot: i ' the the , and the . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 

User: Should I invest in my house, when it's in my wife's name?
FinanceBot: i ' s a to the . . . you can ' t have to the . . . you can ' t have to the . . . you can ' t have to the . . . you can ' t have to the . . . you


User: hello! what is your name?
FinanceBot: i ' s a to the . . . the . . . . . . the . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

---------------------------------------------------------------------------------------

The model is trained for only 20 epochs. A100 GPU by Google 
collab pro was used for model training. It took almost 12 Hours for model training 
but after 13 epochs the runtime was disconnected. All the metrics were 
logged into wandb for tracking and revalidation. The objective from the next 
training is to optimise and enhance model 
performance for faster training. 

------------------------------------------------------------------------------------

The model was not trained on any attention mechanism and with minimal
text pre processing to understand how the model behaves under different scenarios.

-------------------------------------------------------------------------------------

