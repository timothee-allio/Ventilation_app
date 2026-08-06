# 1. Use an official, lightweight Python runtime as a parent image
FROM python:3.12-slim

# 2. Set the working directory in the container
WORKDIR /app

# 3. Copy the requirements file into the container
COPY requirements.txt .

# 4. Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the entire project directory into the container
COPY . .

# 6. Expose the port the app runs on 
# (Note: Change 8050 to 8501 if you are using Streamlit, or whatever port your app binds to)
EXPOSE 8050

# 7. Run the application when the container launches
CMD ["python", "Ventilation_app.py"]