# counselorgpt2.0
Task: 
Augment functionality of CounselorGPT1.0. New chatbot should be able to answer UCLA students' queries about courses that match their interests, student reviews of courses and professors, major/minor requirements, institutional policies, school GE/diversity course requirements, etc. This chatbot should be able to provide students with the same information that an academic counselor would be able to provide.
Implementation:
Step #1: Use Google Chrome's webscraper extension to extract target information from the UCLA course registrar website (https://catalog.registrar.ucla.edu/) and the BruinWalk course review website (https://www.bruinwalk.com/).
Step #2: Perform preliminary data processing in excel.
Step #2: Import .csv file and convert it to a numpy array for further processing.
Step #3: Count the number of tokens in each element of the array. Then partition the elements in the original array among three smaller arrays: one for elements with fewer than 500 tokens, one for elements with 500-1000 tokens, and one for elements with more than 1000 tokens. 
Step #4: To ensure reliable semantic search, need to enforce that all array elements are of similar length (~500-1000 tokens). Write a function that groups elements containing fewer than 500 tokens into aggregates containing ~500 tokens. 
Step #5: Write a second function that splits elements containing more than 1000 tokens into smaller ~500 token chunks. Split large chunks at the " | " character to produce these smaller chunks. To ensure reliable semantic search, append the identifying information at the beginning of each large chunk to the beginnings of its resulting small chunks.
Step #6: Combine all chunks back into a single array. Create embeddings for the elements in this array (i.e. the "knowledge base" array).
Step #7: From pinecone_text.sparse, import BM25Encoder. Use the bm25_encoder's "fit" function to establish tf-idf values for the "knowledge base" corpus. Store these values in a "bm25_values.json" file. Add this file to the Github repository.
Step #8: CounselorGPT1.0 exclusively used semantic similarity to match a students' query to a relevant "document" in the Pinecone database. CounselorGPT2.0, in contrast, uses hybrid search. That is, it finds relevant documents using a combination of keyword search (enabled by the bm25 values) and semantic search. 
Step #9: Create a prompt template for the LLM which tells the model that it is an academic counselor at UCLA and that it should direct students to the course registrar if it doesn't know the answer to a question. 
Step #10: Create a chain interface and pass the prompt template into the "prompt" argument.
Step #11: Call the get_relevant_documents() function of the Pinecone retriever to retrieve documents relevant to the student's query. Then pass those documents, along with the query, into the chain interface. 
Step #12: Access the "output_text" feature of the chain to get the model's response.
Step #13: Deploy on Streamlit and create a chatbot interface, complete with a time-delayed cursor which gives the illusion of the chatbot producing its response one word at a time.

Final product: https://counselorgpt2.streamlit.app/

Application was taken down in July 2023 due to 1) the high cost of maintaining the Pinecone database and 2) rapid changes in the Langchain documentation which rendered my app nonfunctional every few weeks.
