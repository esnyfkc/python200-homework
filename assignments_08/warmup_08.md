# Week 8 Warmup

## Part 1: Cloud Concepts

### Question 1

With cloud computing, you rent resources and pay only for what you use. If you own servers, you pay for the hardware and maintenance even when you are not using all of the capacity.

### Question 2

Vertical scaling means making one machine stronger by adding more CPU, RAM, or a better GPU. Horizontal scaling means adding more machines and splitting the work between them.

I would use vertical scaling if a model training job needs more RAM or a better GPU. I would use horizontal scaling if a website suddenly has too many users for one server.

- **Viral web app:** Horizontal, because the traffic can be spread across multiple servers.
- **Slow model training:** Vertical, because a stronger GPU and more RAM can make the training faster.
- **Pipeline with 10,000 files:** Horizontal, because the files can be processed across multiple machines at the same time.

### Question 3

**Classifications:**

- **Gmail:** SaaS. You just log in and use it, and Google handles everything else.
- **Azure Virtual Machines:** IaaS. You get a virtual machine, but you still need to choose the OS and set up what you need.
- **AWS S3:** IaaS. It gives you cloud storage, and you decide how you want to organize and use it.
- **GitHub Codespaces:** PaaS. GitHub gives you a ready development environment, and you mainly bring your code and configuration.
- **Snowflake:** SaaS. It is a managed data platform, so you can work with your data without setting up and managing servers.
- **Supabase:** BaaS. It gives you things like a database, authentication, storage, and APIs without having to build the whole backend yourself.

**Definitions:**

- **IaaS:** You rent basic cloud resources like virtual machines, storage, and networking. For example, with Azure Virtual Machines, I still manage the OS, software, and updates.
- **PaaS:** The provider manages most of the infrastructure, and I focus more on my code. For example, with Google App Engine, I manage my application and configuration while the platform handles the servers.
- **SaaS:** It is a finished application that you can just use. For example, with Gmail, I mainly manage my account, settings, and data.

### Question 4

A managed data platform like Databricks or Snowflake runs on top of cloud providers and handles a lot of the setup for data work. If you use AWS or GCP directly, you have to choose and connect the services yourself. A managed platform is easier and faster to use, but you give up some flexibility and it can cost more.

### Question 5

The cloud may not be worth it if your data fits on one machine and you do not need much computing power. For a small prototype, running it locally can be simpler and cheaper, and you avoid extra setup and unexpected costs.

# Part 2: Cloud Landscape

## Question 1

- **AWS:** AWS has the widest range of services and is commonly used by startups, large companies, and organizations with engineering teams.
- **Microsoft Azure:** Azure works especially well with Microsoft products, so it is common in large companies, government, and organizations already using Microsoft tools.
- **Google Cloud Platform (GCP):** GCP is especially strong in data analytics and machine learning, so it is a good choice for teams working with data, AI, or ML.

## Question 2

**Reason 1:** Supabase is easier to access. Azure required students to wait for organization setup and authentication, while Supabase lets students create their own account quickly.

**Reason 2:** Supabase is a better fit for learning data work because it uses PostgreSQL. Working with tables, SQL, and relationships is more useful for this course than only storing files.

**Reason 3:** Supabase fits the pipeline better. The raw and enriched data can be stored in two related tables, which makes it easier to check the data and debug problems.

**My reflection:** This shows me that the biggest cloud provider is not always the best choice. I should look at what the project needs, how easy the tool is to use, the cost, and how much extra setup it requires.

## Question 3

1. **10 TB of image files:** Object storage. I could use AWS S3 to store the images and retrieve them by filename.

2. **GPU training job for four hours:** Compute. I could use an AWS EC2 GPU instance, run the training job, and shut it down when I am finished.

3. **Web API that automatically scales:** Serverless compute. I could use AWS Lambda, which can run code without managing a server and scale based on demand.

4. **Send data to an LLM and get text back:** LLM API. I could use Azure OpenAI Service to send the data to a language model and get a response.

## Question 4

**My project:** I could build a small weather project that collects daily weather data, stores it, and processes it to show useful information about each day.

**Stack:**

- **Supabase:** Managed relational database to store the weather data.
- **AWS Lambda:** Serverless compute to run the Python code that processes the data.

Using one provider could make things simpler because I would have one account, one billing system, and services that work well together. But I would give up some flexibility to choose the best tool for each part of the project, and switching providers later could be harder.