# medical_chatbot end to end with AWS deployement

Step 1 - Loading the document using DirectoryLoader and Class PyPDFLoader

step 2 - Cleaned document into minmal metadata

step 3 - splitted those documents into chunks of size 500 each with overlap of 50

step 4 - Embedding those chuncks using Huggin Face Embedding model

step 5 - Store embedded vectore into the vecotre Database - Pinecone with indexex

step 6 - retriving similar chunks using similarity metrics

step 7 - Input from user and then showing the result of quiery
