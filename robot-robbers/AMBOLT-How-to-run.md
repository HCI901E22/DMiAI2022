# Runtime guide for AMBOLT
Here's how to run our robot-robbers on your system:

## Step 1: Pull the directory
Pull The directory from github, it will be made public on october 11 at 14:00

```git clone https://github.com/HCI901E22/DMiAI2022.git KurtsKammerater```

## Step 2: Enter the directory
You should know this one ;-)

```cd KurtsKammerater```

## Step 3: Build the directory
This assumes you have docker installed on your machine, if not install it. 

```docker build -t KK_robots robot-robbers```

NB: you might need to run docker with sudo depending on your setup ¯\_(ツ)_/¯

## Step 4: Run a docker container from the image

```docker run --env-file robot-robbers/environments/dev/.emily.env -p 4343:4343 -d --name KK_robots KK_robots```

## Step 5: Done!
You should now be done and the api should be ready to receive requests on ```http://0.0.0.0:4343/predict```

You might want to do a quick test on ```http://0.0.0.0:4343/api```