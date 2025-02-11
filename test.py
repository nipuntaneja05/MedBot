from pinecone import Pinecone

PINECONE_API_KEY = "pcsk_6hfbZM_PyDhaPVXZs1H5ddoLnsEWx7iNai3SZR8Mfbgsf59yEKteEmXsiatUEu5q2RPML2"  # Replace with your actual API key

pc = Pinecone(api_key=PINECONE_API_KEY)
print(pc.list_indexes())  # List existing indexes
