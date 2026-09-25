class DummyEmbeddings:
    """
    Dummy embedding provider for deterministic tests.
    """

    def embed_documents(self, texts):
        """
        Generate deterministic dummy embeddings for text documents.

        Parameters
        ----------
        texts : list of str
            Input text list.

        Returns
        -------
        list of list of float
            Simple numeric vectors derived from text lengths.
        """
        return [[float(len(text)), 1.0, 0.0] for text in texts]

    def embed_query(self, query):
        """
        Generate a deterministic dummy embedding for a query.

        Parameters
        ----------
        query : str
            Query string.

        Returns
        -------
        list of float
            Simple numeric vector derived from query length.
        """
        return [float(len(query)), 1.0, 0.0]

    def __call__(self, text):
        """
        Make the dummy embedding provider callable for FAISS compatibility.

        Parameters
        ----------
        text : str
            Query text.

        Returns
        -------
        list of float
            Query embedding vector.
        """
        return self.embed_query(text)