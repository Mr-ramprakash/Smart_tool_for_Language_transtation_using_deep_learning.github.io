from flask import Flask, render_template, request, redirect, url_for, session
from flask_socketio import SocketIO, send, emit, join_room, leave_room
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///chat.db'
db = SQLAlchemy(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# User Model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)

# Message Model
class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    sender = db.Column(db.String(80), nullable=False)
    receiver = db.Column(db.String(80), nullable=False)
    content = db.Column(db.String(500), nullable=False)
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())

# Routes
@app.route('/')
def home():
    if 'username' in session:
        return redirect(url_for('chat'))
    return render_template('login.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            session['username'] = username
            return redirect(url_for('chat'))
    return render_template('login.html')

@app.route('/chat')
def chat():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('chat.html', username=session['username'])

# WebSocket Handlers
@socketio.on('message')
def handle_message(data):
    sender = session['username']
    receiver = data['receiver']
    content = data['content']
    
    # Save message to database
    message = Message(sender=sender, receiver=receiver, content=content)
    db.session.add(message)
    db.session.commit()
    
    # Send message to receiver
    emit('new_message', {
        'sender': sender,
        'content': content,
        'timestamp': message.timestamp.strftime("%Y-%m-%d %H:%M:%S")
    }, room=receiver)
    
    # Send message back to sender (for UI update)
    emit('new_message', {
        'sender': sender,
        'content': content,
        'timestamp': message.timestamp.strftime("%Y-%m-%d %H:%M:%S")
    }, room=sender)

@socketio.on('join')
def on_join():
    username = session['username']
    join_room(username)
    emit('status', {'msg': f'{username} has joined the chat'}, room=username)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    socketio.run(app, debug=True)