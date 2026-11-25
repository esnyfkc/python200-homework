# Week 8 Project

## Supabase Setup

My Supabase project is set up, and both `weather_raw` and `weather_enriched` tables were created successfully with the required columns.

## Cloud Cost Analysis

Scenario A costs about **$1.66 per month** for a `t3.micro` instance running 160 hours per month.

Scenario B costs about **$2,398.79 per month**. Most of the cost comes from the `p3.2xlarge` GPU instance at **$2,233.80 per month**. The RDS database costs about **$141.44 per month**, and 1 TB of S3 Standard storage costs about **$23.55 per month**.

I was surprised by how much more expensive the GPU instance was compared with the lightweight compute. It shows that a GPU instance can be useful when a job really needs that much computing power, but it would be very expensive to keep running when it is not needed.

For the RDS estimate, I used PostgreSQL with a Single-AZ deployment, On-Demand pricing, and 100 GB of General Purpose SSD storage because the assignment did not specify those settings.

## Video

Video link: https://drive.google.com/file/d/1BiRfLQkSM9QgBNKBSPl-xhPNwM0mHbmY/view?usp=sharing