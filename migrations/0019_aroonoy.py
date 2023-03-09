# Generated by Django 4.1.5 on 2023-01-31 18:21

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('marketBotApp', '0018_rename_buy_point_optimal_buy_sell_points_value_and_more'),
    ]

    operations = [
        migrations.CreateModel(
            name='AroonOY',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('symbol', models.CharField(max_length=10)),
                ('period', models.IntegerField()),
                ('aroon_up', models.FloatField()),
                ('aroon_down', models.FloatField()),
                ('stock', models.ForeignKey(default=None, on_delete=django.db.models.deletion.CASCADE, to='marketBotApp.finnish_stock_daily')),
            ],
        ),
    ]