# PaLM Bartender - your personal bartender

This python script uses Google PaLM Text API, Imagen API and Langchain to build an application that serves cocktail drinks.

Google's LLM models support the app with content generation; Langchain libraries such as MultiPromptChain and ConversationBufferMemory support routing of the queries and retaining conversation history; Streamlit supports app framework for front-end presentation.

## Installation
To install the dependencies for this script, you can use the following command:

pip install -r requirements.txt

## Authentication
To authenticate against your Google Cloud project, you can use the following command:

gcloud auth application-default set-quota-project "your-project-id"

## Usage

To run the script, you can use the following command:

streamlit run palm_bartender.py or python -m streamlit run palm_bartender.py

This will open the app in your web browser. You can then interact with the app by ordering a cocktail or getting a cocktail recipe, the specific features are:

**Order a cocktail** - it's preferable for you to describe the type of cocktail and particular notes you're looking for, in return palm_bartender will serve you a perfect cocktail and show you a visual illustration while also providing you food recommendations.

**Get a cocktail recipe** - if you have liquors in your drink cabinet and ingredients in your fridge, you can ask palm_bartender to recommend a cocktail recipe using mainly those that you have, it provides specific instructions on how to make that cocktail by yourself

## Misc
Hope you like the PaLM Bartender app, and let me know if you have any suggestions to improve the experience:)