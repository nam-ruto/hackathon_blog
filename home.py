import streamlit as st

def inspiration():
    st.subheader("🪴Inspiration", divider=True)
    st.markdown("""Every project starts with a spark, and for us, it was seeing how students struggle to navigate the wealth of information in our college catalog. 
                We noticed that many students felt overwhelmed when trying to decide which courses to take or figure out degree requirements. 
                We wanted to build a tool that could make academic planning easier, more accessible, and faster—hence the idea of a Virtual College Consultant chatbot was born. 
                Our goal was simple: to create an intelligent assistant that guides students through the maze of academic options.""")
    
def what_it_does():
    st.subheader("🎯What it does", divider=True)
    st.markdown("""The **Troy Virtual College Consultant chatbot** is designed to help Troy's students by providing:""")
    st.markdown("* **Personalized Course Recommendations**: Students can ask about courses related to their interests or majors, and the chatbot will suggest relevant options.")
    st.markdown("* **Degree Program Information**: It provides detailed information about degree requirements, electives, and core subjects.")
    st.markdown("* **Campus Resources Guidance**: Whether it’s help finding the right advisor, scholarships, or campus facilities, the chatbot guides students in the right direction.")
    st.markdown("* **Real-time Answers**: The chatbot retrieves the most relevant information from the college catalog instantly, helping students get answers quickly without scrolling through pages of documents.")
    
def how_we_built():
    st.subheader("✅How We Built It", divider=True)
    st.markdown("""
                We implemented the chatbot using a technique called Retrieval-Augmented Generation (RAG), which combines information retrieval with natural language generation to provide accurate and context-aware responses. 
                The development process involved multiple stages, from data extraction to AI model integration. 
                Here's a breakdown of the key components:
                """)
    st.markdown("* **Data Extraction and Preprocessing**: The first step was to extract relevant academic data from the college's PDF catalog. PDFs can be challenging due to their unstructured nature, so we used Python libraries like PyPDF2, pdfplumber, PyMuPDF to parse the text and identify key sections, such as course descriptions, program requirements, and academic resources.")
    st.markdown("* **Data Embedding (Vectorization)**: To enable efficient retrieval, we transformed the catalog data into vector embeddings using all-MiniLM-L6-v2, which is a pre-trained language model optimized for embedding text into fixed-length vectors. Each course description, program requirement, or resource entry was embedded into a vector, which allows the chatbot to semantically search for and retrieve the most relevant information when a user asks a question.")
    st.markdown("* **ChromaDB for Retrieval**: We stored the vector embeddings in ChromaDB, a vector database designed for fast, similarity-based retrieval. This database allows us to quickly search for catalog entries that closely match the user's query.")
    st.markdown("* **Generative AI Model**: Once the relevant data is retrieved, we use OpenAI's GPT model to generate a natural language response. The GPT model takes the user's query and the retrieved catalog entries as inputs, and then formulates a coherent, personalized response.")
    st.markdown("* **Backend with FastAPI**: We built the backend using FastAPI, a modern web framework for building APIs with Python. FastAPI handles user requests and manages the interaction between the chatbot interface, the vector database, and the generative model")
    st.markdown("* **Frontend with Flask**: For the user interface, we opted for Flask, a lightweight and flexible web framework for building dynamic web applications. Flask provides more control over the design and structure of the application, allowing us to create a custom and responsive interface for users to interact with the chatbot.")

def challenge():
    st.subheader("⚠️Challenges We Ran Into", divider=True)
    st.markdown("""
                One of the major challenges we faced was in the data aggregation phase. Initially, we attempted to crawl data directly from the Troy's website and Canvas (the reason behind is there are more useful information in these platform). 
                However, after sending around 100 requests to the website, we hit a rate limit and got blocked from further crawling. 
                This forced us to reconsider our approach🤔.
                """)
    st.markdown("""
                To overcome this, we decided to extracting the data from the college's catalog PDF file. 
                Working with PDF files also posed its own challenges, as they are unstructured and difficult to parse. 
                We had to use Python libraries like PyMuPDF and pdfplumber to extract key data points such as course descriptions and program details, and then clean and organize this data into a usable format.
                """)

def what_next():
    st.subheader("💡What's next", divider=True)
    st.markdown("We have a few exciting ideas for the future:")
    st.markdown("""
                * **Broaden the data sources**: In the future, we want to include additional data like career services, scholarship opportunities, and campus events.
                * **User personalization**: Imagine if the chatbot could suggest courses based on a student’s academic history or recommend activities that align with their interests.
                * **Expand language support**: Making the chatbot multi-lingual would allow us to better serve international students, providing guidance in their native language.
                * **AI-driven insights**: We’d love to use predictive AI to help students choose courses they are likely to excel in based on their strengths.
                """)

def try_it_out():
    st.subheader("✅Try it out", divider=True)
    st.markdown("""You can try the **beta version** here in this blog. Navigate to the sidebar and input the ```OpenAI API Key``` before starting to chat with the chatbot""")
    st.info("🪴Because this is the beta version you have to use your own API key to use this service")
    st.image("assets/landingPage.jpg", caption="Landing page")
    st.image("assets/chatbot_beta.png", caption="Chatbot")

def showAll():
    st.header("🧑🏻‍🎓TroyAI Advisor Chatbot")
    st.markdown("""\n""")
    inspiration()
    what_it_does()
    how_we_built()
    challenge()
    what_next()
    try_it_out()


showAll()
    