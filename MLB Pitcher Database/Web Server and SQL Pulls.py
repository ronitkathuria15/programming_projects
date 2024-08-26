import os
from sqlalchemy import *
from sqlalchemy.pool import NullPool
from flask import Flask, request, render_template, g, redirect, Response

tmpl_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
app = Flask(__name__, template_folder=tmpl_dir)

DATABASE_USERNAME = "patrick.jiao"
DATABASE_PASSWRD = "1299"
DATABASE_HOST = "34.28.53.86" 
DATABASEURI = f"postgresql://{DATABASE_USERNAME}:{DATABASE_PASSWRD}@{DATABASE_HOST}/project1"

engine = create_engine(DATABASEURI)



@app.before_request
def before_request():
	try:
		g.conn = engine.connect()
	except:
		print("uh oh, problem connecting to database")
		import traceback; traceback.print_exc()
		g.conn = None

@app.teardown_request
def teardown_request(exception):
	try:
		g.conn.close()
	except Exception as e:
		pass


@app.route('/teams', methods=['GET', 'POST'])
def filter_teams():
    if request.method == 'POST':
        # get the form data
        city = request.form.get('city')
        league = request.form.get('league')
        playoffs_id = request.form.get('playoffs_id')
        
        # build the SQL query based on the form data
        sql_query = "SELECT * FROM team"
        if city:
            sql_query += " WHERE city = '{}'".format(city)
        if league:
            sql_query += " WHERE league = '{}'".format(league)
        if playoffs_id:
            sql_query += " WHERE playoffs_id = '{}'".format(playoffs_id)
        sql_query += ";"
        
        # execute the query and get the results
        cursor = g.conn.execute(text(sql_query))
        teams = cursor.fetchall()
        return render_template('teams.html', teams=teams)
    else:
        # get all the teams from the database
        cursor = g.conn.execute(text("SELECT * FROM team"))
        teams = cursor.fetchall()
        return render_template('teams.html', teams=teams)
    

@app.route('/stats', methods=['GET', 'POST'])
def filter_stats():
    if request.method == 'POST':
        # get the form data
        sort_by = request.form.get('sort_by')
        
        # build the SQL query based on the form data, joining the `pitcher` and `statistics` tables
        sql_query = "SELECT * FROM pitcher JOIN statistics ON pitcher.statistics_id = statistics.statistics_id JOIN all_star ON pitcher.all_star_id = all_star.all_star_id"
        if sort_by:
            sql_query += f" ORDER BY {sort_by} DESC"
        sql_query += ";"
        
        # execute the query and get the results
        cursor = g.conn.execute(text(sql_query))
        results = cursor.fetchall()
        return render_template('stats.html', results=results)
    else:
        # get all the pitchers from the database
        cursor = g.conn.execute(text("SELECT * FROM pitcher JOIN statistics ON pitcher.statistics_id = statistics.statistics_id JOIN all_star ON pitcher.all_star_id = all_star.all_star_id"))
        results = cursor.fetchall()
        return render_template('stats.html', results=results)

@app.route('/pitcher', methods=['GET', 'POST'])
def filter_pitchers():
    if request.method == 'POST':
        # Get form data
        country = request.form.get('country_name')
        sort_by = request.form.get('sort_by')
        sort_order = request.form.get('order')
        has_injury = request.form.get('has_injury')
        hand = request.form.get('pitching_hand')
        
        # Build SQL query and join the required tables for filtering
        sql_query = "SELECT * FROM pitcher JOIN contract ON pitcher.pitcher_id = contract.pitcher_id JOIN country ON pitcher.country_name = country.country_name JOIN on pitcher.injury_id = injury.injury_id"
        if country:
            sql_query += " WHERE country_name = '{}'".format(country)
        if hand:
            sql_query += " WHERE pitching_hand = '{}'".format(hand)
        if has_injury:
            sql_query += " WHERE injury_id IS NOT NULL".format(has_injury)
        sql_query += ";"
        
        # Add sorting and ordering
        if sort_by:
            sql_query += f" ORDER BY {sort_by} {sort_order}"
        sql_query += ";"
        
        # Execute the query and get the results
        cursor = g.conn.execute(text(sql_query))
        pitchers = cursor.fetchall()
        return render_template('pitcher.html', pitchers=pitchers)
    else:
        # Get all pitchers from the database
        cursor = g.conn.execute(text("SELECT * FROM pitcher JOIN contract ON pitcher.pitcher_id = contract.pitcher_id JOIN country ON pitcher.country_name = country.country_name JOIN on pitcher.injury_id = injury.injury_id"))
        pitchers = cursor.fetchall()
        return render_template('pitcher.html', pitchers=pitchers)


@app.route('/login')
def login():
	abort(401)
	this_is_never_executed()

@app.route('/index')
def homeindex():
        return render_template("index.html")

if __name__ == "__main__":
	import click

	@click.command()
	@click.option('--debug', is_flag=True)
	@click.option('--threaded', is_flag=True)
	@click.argument('HOST', default='0.0.0.0')
	@click.argument('PORT', default=8111, type=int)
	def run(debug, threaded, host, port):
		"""
		This function handles command line parameters.
		Run the server using:

			python server.py

		Show the help text using:

			python server.py --help

		"""

		HOST, PORT = host, port
		print("running on %s:%d" % (HOST, PORT))
		app.run(host=HOST, port=PORT, debug=debug, threaded=threaded)

run()