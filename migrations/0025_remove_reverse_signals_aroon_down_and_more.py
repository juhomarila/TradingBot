# Generated by Django 4.1.5 on 2023-02-03 19:22

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('marketBotApp', '0024_reverse_signals'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='reverse_signals',
            name='aroon_down',
        ),
        migrations.RemoveField(
            model_name='reverse_signals',
            name='aroon_up',
        ),
    ]