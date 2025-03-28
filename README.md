# ChatbotV2-Her2


This project is developed as a part of an assignment: 

We need to develop a Q/A chatbot prototype that can effectively address questions related to this publication. This prototype should be implemented in Python using an open-source Large Language Model (LLM). Additionally, we require a comprehensive evaluation approach to assess the chatbot's performance.


# How to install the Chatbot:

Need python environment >= 3.11.10

Please clone the repo:
<pip install -r requirements.txt>

Please use streamlit to run the chatbot:

< streamlit run main.py>


# Chatbot Architecture

![image](https://github.com/user-attachments/assets/7a54b265-1f53-480f-a50d-e1249911d5f3)


The model is fine-tuned version of ContactDoctor/Bio-Medical-LLama-3-2-1B-CoT 

It gave the best performance across different metrics of accuracy, bleu score, model loss

Fine-tuning was done using Peft QLora. Link to the colab notebook [link](https://colab.research.google.com/drive/1g10xCCYbDjQz2e0j19BQMCmfgNbstSAq#scrollTo=hA_DnlE97gXW)




citation: 
@misc{ContactDoctor_Bio-Medical-Llama-3.2-1B-CoT-012025,
  author = {ContactDoctor},
  title = {Bio-Medical-Llama-3-2-1B-CoT-012025: A Reasoning-Enhanced Biomedical Language Model},
  year = {2025},
  howpublished = {https://huggingface.co/ContactDoctor/Bio-Medical-Llama-3-2-1B-CoT-012025},
}
