FROM python:3.10

COPY . .

# Install Jupyter
RUN pip install --no-cache-dir -r requirements.txt

# Set up a working directory
WORKDIR /app

# Expose the port Jupyter will run on
EXPOSE 8888

# Run Jupyter Notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]
