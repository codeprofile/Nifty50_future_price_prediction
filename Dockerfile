
FROM ubuntu:latest
RUN apt-get update -y
RUN apt-get install -y python-pip python-dev build-essential

RUN pip3 install -r requirements.txt
#Set the working directory
WORKDIR /Forecast_trends/

#copy all the files
COPY . .



#Expose the required port
EXPOSE 5000

#Run the command
CMD ["gunicorn", "main:app"]
