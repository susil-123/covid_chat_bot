import streamlit as st
import random
import json
from PIL import Image
import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
st.set_page_config(layout="wide")
st.markdown("""
<style>
.big-font {
    font-size:100px !important;
}
.small-font{
    font-size:20px;
}
</style>
""", unsafe_allow_html=True)


st.markdown('<p class="big-font">Covid chatbot using deep learning</p>', unsafe_allow_html=True)

img=Image.open('bot.jpg')
st.image(img,use_column_width=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "sam"
# print("Let's chat! (type 'quit' to exit)")
# st.markdown('<p class="small-font">Let\'s chat! (type \'quit\' to exit)</p>', unsafe_allow_html=True)
st.subheader("Let\'s chat!")
sentense=st.text_area("")
while (sentense):
    # sentence = "do you use credit cards?"
    sentence = str(sentense)
    # sentence =input("You: ")
    if sentence == "quit":
        # print(f"{bot_name}: bye then")
        st.write(f"{bot_name}: bye then")
        break

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                # print(f"{bot_name}: {random.choice(intent['responses'])}")
                st.write(f"{bot_name}: {random.choice(intent['responses'])}")
    
    else:
        # print(f"{bot_name}: I do not understand...")
        st.write(f"{bot_name}: I do not understand...")
    break