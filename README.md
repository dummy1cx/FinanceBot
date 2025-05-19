# FinanceBot
Finance industry faces several challenges in solving customer queries as it requires a professional level
of domain knowledge and each problem can be different according to customer’s need. The customer
service officer needs to be proficient with simple financial terms like LTV, diversification, hedging to
complex aspects of revenue modeling and this requires Financial firms to spend millions in training and
upskilling their employees but that does not guarantee giving the best resolutions as per customer needs
and requirements.

To solve this problem our project tries to develop an advanced chatbot which not only can understand
customer’s problems but also gives personalized feedback to help consumers’ financial behaviour. It will
improve both customer’s user experience and operational efficiency of firms as it requires less manpower.
The developed chatbot can be deployed for live applications to help customers 24X7. It will help users to
make better decisions in their financial decision making, helping them to improve their financial literacy
while maintaining strict regulatory compliances recommended by competent authorities.
To develop the chatbot which we named as “FinanceBot” this project explored various advanced
architectures to improve the accuracy and performance in advanced financial question answering. To
train the model, this project used a simple Sequence to Sequence(LSTM) architecture without any
attention mechanisms as a baseline experiments, the results are unsatisfactory hence for the second
experiment we continued our Sequence to Sequence(GRU) training with adding a Luong attention in
the architecture, the results improved but still it was not acceptable. For the final Sequence to Sequence
model training we have integrated a pre trained Glove embedding finetuned with our custom dataset
along with Luong attention. We replaced the GRU cells from the architecture with LSTM cell blocks,
observing a substantial improvement in model performance and managed to achieve a perplexity score of
2 (approx). Furthermore we continued our model training with Transformers architecture for advanced
model training. The first training was conducted with transformers along with multi head attention but
perplexity was achieved around 15. Furthermore The training was conducted with Talking head attention
and MQA which brought down the perplexity to 2.63 and 1.20 respectively. With MQA we able to
achieve a score which is almost closer to 1 which was satisfactory.

Though out of all the experiments we have achieved a good perplexity score but still it was noticed
that the model is unable to give satisfactory answers. Hence the model needs further experiments with
varied and large datasets to understand the context which we kept as a future part of the project. In
addition to that, we also plan to develop a fully optimised and stacked transformers architecture with
multiple attention heads like GPT-2 to train the model. Out of all the experiment conducted above proves
that changing the architecture with advanced transformers along with a large dataset can improve the
performance of Chatbot before deploying it for pro
