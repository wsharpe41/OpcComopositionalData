import psycopg2

import configparser

config = configparser.ConfigParser()
config.read('database.ini')

host = config.get('postgresql', 'host')
database = config.get('postgresql', 'database')
user = config.get('postgresql', 'user')
password = config.get('postgresql', 'password')
port = config.get('postgresql', 'port')

conn = psycopg2.connect(
    host=host,
    database=database,
    user=user,
    password=password,
    port=port
)

c = conn.cursor()

# create database if it doesn't exist
c.execute("SELECT 1 FROM pg_database WHERE datname = 'opc'")
result = c.fetchone()

if not result:
    # Create the database if it does not exist
    c.execute("CREATE DATABASE opc;")


# create table aerosol if it doesn't exist
c.execute('''CREATE TABLE IF NOT EXISTS aerosol
                (aerosol_id SERIAL PRIMARY KEY,
                gsd REAL,
                gm REAL,
                par_counts REAL,
                svp REAL,
                svp_ref REAL,
                mm REAL,
                kappa REAL,
                hvap REAL,
                density REAL
                )''')

# create table bin_counts if it doesn't exist
c.execute('''CREATE TABLE IF NOT EXISTS bin_counts
                (bin_id SERIAL PRIMARY KEY,
                rh REAL,
                temp REAL,
                bin_0 REAL,
                bin_1 REAL,
                bin_2 REAL,
                bin_3 REAL,
                bin_4 REAL,
                bin_5 REAL,
                bin_6 REAL,
                bin_7 REAL,
                bin_8 REAL,
                bin_9 REAL,
                bin_10 REAL,
                bin_11 REAL,
                bin_12 REAL,
                bin_13 REAL,
                bin_14 REAL,
                bin_15 REAL,
                bin_16 REAL,
                bin_17 REAL,
                bin_18 REAL,
                bin_19 REAL,
                bin_20 REAL,
                bin_21 REAL,
                bin_22 REAL,
                bin_23 REAL
                )''')

# create table experiments if it doesn't exist
c.execute('''CREATE TABLE IF NOT EXISTS experiments
                (experiment_id SERIAL PRIMARY KEY,
                flow_one integer,
                flow_two integer,
                flow_three integer,
                flow_four integer,
                flow_five integer,
                category TEXT)''')


# Add many to many relationship between experiments and aerosol
c.execute('''CREATE TABLE IF NOT EXISTS aerosol_experiments
                (aerosol_id integer,
                experiment_id integer,
                FOREIGN KEY (aerosol_id) REFERENCES aerosol(aerosol_id),
                FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id))''')

# commit changes to database
conn.commit()
# close connection to database
conn.close()
