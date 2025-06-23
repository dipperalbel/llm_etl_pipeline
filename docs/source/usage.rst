Usage
==========================

To use this project, you will primarily interact with its `__main__` file from your terminal.

Running the Extraction Process
----------------

1.  **Ensure your Ollama server is running** and the required models (`phi4:14b` and `gemma3:27b`) are pulled.
2.  **Activate your Poetry shell** (if you haven't already):
    ```bash
    poetry shell
    ```
3.  **Execute the main script:**
    ```bash
    poetry run python main.py 
    ```
    
4.  **Provide the PDF directory path:**
    The script will prompt you to enter the path to the directory containing the PDF documents.
    ```
    Please enter the path to the directory containing the PDF documents: 
    ```
    You should enter the full path to your PDF folder, for example:
    * **On Windows:** `C:\path\to\your\pdf_documents`
    * **On Linux/macOS:** `/home/user/path/to/your/pdf_documents`

MAKE SURE THAT THE INPUT FOLDER CONTAINS THE CALL FOR PROPOSAL PDFS (the PDFs should be like AMIF-2024-TF2-AG-INFO-01_separator_call-...).