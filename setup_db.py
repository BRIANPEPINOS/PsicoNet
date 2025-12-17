from app import app, db, User
from werkzeug.security import generate_password_hash

with app.app_context():
    db.create_all()

 
    if not User.query.filter_by(username='admin').first():
        admin = User(
            username='admin',
            password_hash=generate_password_hash('admin123'), # Contraseña del admin
            name='Administrador Principal',
            role='admin'
        )
        db.session.add(admin)
        print("Administrador creado.")

    
    students = [
        ('brian', 'brian123', 'Brian Pepinos'),
        ('edith', 'edith123', 'Edith Chuico'),
        ('freddy', 'freddy123', 'Freddy Jimenez'),
        ('elian', 'elian123', 'Elian Zambrano'),
        ('adonis', 'adonis123', 'Adonis Alegria'),
        ('aldo', 'aldo123', 'Aldo Saula'),
        ('joseph', 'joseph123', 'Joseph Franco'),
        ('helem', 'helem123', 'Helem Quintero'),
        ('carlos', 'carlos123', 'Carlos Calapucha'),
        ('jennifert', 'jennifert123', 'Jennifer Torres'),
        ('miguel', 'miguel123', 'Miguel Molina'),
        ('anderson', 'anderson123', 'Lara Anderson'),
        ('ronald', 'ronald123', 'Ronald Puruncajas'),
        ('juan', 'juan123', 'Juan Jimenez'),
        ('pamela', 'pamela123', 'Pamela Tapia'),
        ('kevin', 'kevin123', 'Kevin Chuquitarco')
        
    ]

    for user, pwd, real_name in students:
        if not User.query.filter_by(username=user).first():
            new_student = User(
                username=user,
                password_hash=generate_password_hash(pwd),
                name=real_name,
                role='user'
            )
            db.session.add(new_student)
            print(f"Estudiante {real_name} creado.")

    db.session.commit()
    print("Base de datos configurada exitosamente.")