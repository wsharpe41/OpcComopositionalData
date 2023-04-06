import sqlite3


conn = sqlite3.connect('../../opc.db')
c = conn.cursor()

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
                bin0 REAL,
                bin1 REAL,
                bin2 REAL,
                bin3 REAL,
                bin4 REAL,
                bin5 REAL,
                bin6 REAL,
                bin7 REAL,
                bin8 REAL,
                bin9 REAL,
                bin10 REAL,
                bin11 REAL,
                bin12 REAL,
                bin13 REAL,
                bin14 REAL,
                bin15 REAL,
                bin16 REAL,
                bin17 REAL,
                bin18 REAL,
                bin19 REAL,
                bin20 REAL,
                bin21 REAL,
                bin22 REAL,
                bin23 REAL
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
