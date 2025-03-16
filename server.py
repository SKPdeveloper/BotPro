#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Серверна частина Telegram-бота для автоматичного наповнення та просування каналу.
"""

import os
import sys
import json
import time
import logging
import sqlite3
import asyncio
import datetime
import traceback
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from telethon import TelegramClient, events
from telethon.tl.functions.messages import GetHistoryRequest
from telethon.tl.functions.channels import GetFullChannelRequest, JoinChannelRequest
from telethon.tl.types import InputChannel, InputPeerChannel, PeerChannel
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Налаштування логування
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("server.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Конфігураційні змінні з наданими даними
API_ID = 21156627
API_HASH = "6eab1d1c790d1629d9ae5fe1ca74530f"
SESSION_NAME = "tg_bot_session"
CHANNEL_ID = None  # Буде встановлено під час запуску
DATABASE_PATH = "tg_bot.db"

# Відображення інформації про конфігурацію
logger.info("------------------------")
logger.info("Конфігурація сервера:")
logger.info(f"API_ID: {API_ID}")
logger.info(f"API_HASH: {API_HASH}")
logger.info(f"SESSION_NAME: {SESSION_NAME}")
logger.info("------------------------")

class TelegramServer:
    """Клас для роботи з Telegram API та обробки даних"""
    
    def __init__(self, api_id, api_hash, session_name, channel_id):
        """
        Ініціалізація серверної частини
        
        :param api_id: API ID для Telegram
        :param api_hash: API Hash для Telegram
        :param session_name: Ім'я сесії
        :param channel_id: ID цільового каналу
        """
        self.api_id = api_id
        self.api_hash = api_hash
        self.session_name = session_name
        self.channel_id = channel_id
        self.client = None
        self.channel = None
        self.db_conn = None
        
        logger.info("Ініціалізація серверної частини бота")
        logger.info(f"Цільовий канал ID: {channel_id}")
        
        # Ініціалізація бази даних
        self._init_database()
        
    async def connect(self):
        """З'єднання з Telegram API"""
        logger.info("Спроба з'єднання з Telegram API")
        
        try:
            self.client = TelegramClient(self.session_name, self.api_id, self.api_hash)
            await self.client.start()
            
            # Отримання інформації про канал
            if self.channel_id.isdigit():
                # Якщо ID числовий, використовуємо його безпосередньо
                entity = await self.client.get_entity(PeerChannel(int(self.channel_id)))
            else:
                # Якщо ID не числовий, це може бути username каналу
                entity = await self.client.get_entity(self.channel_id)
            
            self.channel = entity
            
            logger.info(f"Успішне з'єднання з Telegram API")
            logger.info(f"Цільовий канал: {self.channel.title} (ID: {self.channel.id})")
            
            return True
        except Exception as e:
            logger.error(f"Помилка з'єднання з Telegram API: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def _init_database(self):
        """Ініціалізація бази даних"""
        logger.info("Ініціалізація бази даних")
        
        try:
            if not os.path.exists(DATABASE_PATH):
                logger.info("База даних не знайдена. Створення нової бази даних...")
            
            self.db_conn = sqlite3.connect(DATABASE_PATH)
            cursor = self.db_conn.cursor()
            
            # Створення таблиць
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS posts (
                    id INTEGER PRIMARY KEY,
                    message_id INTEGER,
                    date TEXT,
                    text TEXT,
                    views INTEGER DEFAULT 0,
                    forwards INTEGER DEFAULT 0
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS reactions (
                    id INTEGER PRIMARY KEY,
                    message_id INTEGER,
                    reaction TEXT,
                    count INTEGER,
                    FOREIGN KEY (message_id) REFERENCES posts(message_id)
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS audience (
                    id INTEGER PRIMARY KEY,
                    date TEXT,
                    subscribers INTEGER
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS scheduled_posts (
                    id INTEGER PRIMARY KEY,
                    scheduled_time TEXT,
                    post_text TEXT,
                    media_path TEXT,
                    buttons TEXT,
                    status TEXT DEFAULT 'pending'
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS relevant_channels (
                    id INTEGER PRIMARY KEY,
                    channel_id INTEGER,
                    channel_title TEXT,
                    subscribers INTEGER,
                    relevance_score REAL
                )
            ''')
            
            self.db_conn.commit()
            logger.info("База даних успішно ініціалізована")
            
        except Exception as e:
            logger.error(f"Помилка при ініціалізації бази даних: {str(e)}")
            logger.error(traceback.format_exc())
    
    async def collect_channel_stats(self):
        """Збір статистики каналу"""
        logger.info("Початок збору статистики каналу")
        
        try:
            # Отримання інформації про канал
            full_channel = await self.client(GetFullChannelRequest(channel=self.channel))
            
            # Збереження поточної кількості підписників
            subscribers = full_channel.full_chat.participants_count
            today = datetime.datetime.now().strftime("%Y-%m-%d")
            
            cursor = self.db_conn.cursor()
            cursor.execute("INSERT INTO audience (date, subscribers) VALUES (?, ?)", (today, subscribers))
            self.db_conn.commit()
            
            logger.info(f"Поточна кількість підписників: {subscribers}")
            
            # Отримання останніх повідомлень
            offset_id = 0
            limit = 100
            all_messages = []

            while True:
                logger.info(f"Отримання повідомлень (offset: {offset_id}, limit: {limit})")
                
                history = await self.client(GetHistoryRequest(
                    peer=self.channel,
                    offset_id=offset_id,
                    offset_date=None,
                    add_offset=0,
                    limit=limit,
                    max_id=0,
                    min_id=0,
                    hash=0
                ))
                
                if not history.messages:
                    break
                
                messages = history.messages
                all_messages.extend(messages)
                
                offset_id = messages[-1].id
                
                if len(messages) < limit:
                    break
            
            logger.info(f"Отримано всього {len(all_messages)} повідомлень")
            
            # Обробка та збереження повідомлень
            for message in all_messages:
                cursor.execute(
                    "SELECT id FROM posts WHERE message_id = ?",
                    (message.id,)
                )
                
                if cursor.fetchone() is None:
                    # Додаємо нове повідомлення
                    date_str = message.date.strftime("%Y-%m-%d %H:%M:%S")
                    text = message.message if hasattr(message, 'message') else ""
                    views = message.views if hasattr(message, 'views') else 0
                    forwards = message.forwards if hasattr(message, 'forwards') else 0
                    
                    cursor.execute(
                        "INSERT INTO posts (message_id, date, text, views, forwards) VALUES (?, ?, ?, ?, ?)",
                        (message.id, date_str, text, views, forwards)
                    )
                    
                    logger.info(f"Додано нове повідомлення ID: {message.id}, Перегляди: {views}")
                else:
                    # Оновлюємо статистику
                    views = message.views if hasattr(message, 'views') else 0
                    forwards = message.forwards if hasattr(message, 'forwards') else 0
                    
                    cursor.execute(
                        "UPDATE posts SET views = ?, forwards = ? WHERE message_id = ?",
                        (views, forwards, message.id)
                    )
                    
                    logger.info(f"Оновлено статистику для повідомлення ID: {message.id}, Перегляди: {views}")
                
                # Збереження реакцій
                if hasattr(message, 'reactions') and message.reactions:
                    for reaction in message.reactions.results:
                        cursor.execute(
                            "INSERT OR REPLACE INTO reactions (message_id, reaction, count) VALUES (?, ?, ?)",
                            (message.id, reaction.reaction.emoticon, reaction.count)
                        )
                        
                        logger.info(f"Реакція на повідомлення {message.id}: {reaction.reaction.emoticon} - {reaction.count}")
            
            self.db_conn.commit()
            logger.info("Статистика каналу успішно зібрана")
            
        except Exception as e:
            logger.error(f"Помилка при зборі статистики каналу: {str(e)}")
            logger.error(traceback.format_exc())
    
    async def analyze_emoji_reactions(self):
        """Аналіз емодзі-реакцій на пости"""
        logger.info("Початок аналізу емодзі-реакцій")
        
        try:
            cursor = self.db_conn.cursor()
            cursor.execute("""
                SELECT p.message_id, p.text, r.reaction, r.count, p.views
                FROM posts p
                JOIN reactions r ON p.message_id = r.message_id
                ORDER BY p.date DESC
            """)
            
            rows = cursor.fetchall()
            
            if not rows:
                logger.info("Немає даних для аналізу емодзі-реакцій")
                return None
            
            logger.info(f"Отримано {len(rows)} записів для аналізу емодзі")
            
            # Створення DataFrame для аналізу
            df = pd.DataFrame(rows, columns=['message_id', 'text', 'reaction', 'count', 'views'])
            
            # Аналіз найпопулярніших емодзі
            popular_emoji = df.groupby('reaction')['count'].sum().sort_values(ascending=False)
            logger.info("Найпопулярніші емодзі:")
            for emoji, count in popular_emoji.items()[:5]:
                logger.info(f"{emoji}: {count}")
            
            # Аналіз емодзі відносно переглядів
            df['reaction_rate'] = df['count'] / df['views'] * 100
            avg_reaction_rate = df.groupby('reaction')['reaction_rate'].mean().sort_values(ascending=False)
            
            logger.info("Середній відсоток реакцій відносно переглядів:")
            for emoji, rate in avg_reaction_rate.items()[:5]:
                logger.info(f"{emoji}: {rate:.2f}%")
            
            # Створення графіку
            plt.figure(figsize=(10, 6))
            
            # Вибір топ-10 емодзі для відображення
            top_emoji = popular_emoji.head(10)
            
            sns.barplot(x=top_emoji.index, y=top_emoji.values)
            plt.title('Топ-10 популярних емодзі')
            plt.xlabel('Емодзі')
            plt.ylabel('Кількість реакцій')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Збереження графіку
            buf = BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            
            logger.info("Аналіз емодзі-реакцій успішно завершено")
            
            return {
                'popular_emoji': popular_emoji.to_dict(),
                'avg_reaction_rate': avg_reaction_rate.to_dict(),
                'plot': buf
            }
            
        except Exception as e:
            logger.error(f"Помилка при аналізі емодзі-реакцій: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    async def analyze_audience_growth(self):
        """Аналіз приросту аудиторії"""
        logger.info("Початок аналізу приросту аудиторії")
        
        try:
            cursor = self.db_conn.cursor()
            cursor.execute("SELECT date, subscribers FROM audience ORDER BY date")
            
            rows = cursor.fetchall()
            
            if not rows:
                logger.info("Немає даних для аналізу приросту аудиторії")
                return None
            
            logger.info(f"Отримано {len(rows)} записів про аудиторію")
            
            # Створення DataFrame для аналізу
            df = pd.DataFrame(rows, columns=['date', 'subscribers'])
            df['date'] = pd.to_datetime(df['date'])
            
            # Розрахунок щоденного приросту
            df['growth'] = df['subscribers'].diff()
            
            # Створення графіку
            plt.figure(figsize=(12, 6))
            
            # Графік кількості підписників
            plt.subplot(2, 1, 1)
            sns.lineplot(x='date', y='subscribers', data=df, marker='o')
            plt.title('Динаміка аудиторії каналу')
            plt.xlabel('Дата')
            plt.ylabel('Кількість підписників')
            
            # Графік щоденного приросту
            plt.subplot(2, 1, 2)
            sns.barplot(x=df['date'].dt.strftime('%Y-%m-%d').tolist(), y=df['growth'].fillna(0).tolist())
            plt.title('Щоденний приріст підписників')
            plt.xlabel('Дата')
            plt.ylabel('Приріст')
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            
            # Збереження графіку
            buf = BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            
            logger.info("Аналіз приросту аудиторії успішно завершено")
            
            # Розрахунок статистики
            current_subscribers = df['subscribers'].iloc[-1]
            previous_subscribers = df['subscribers'].iloc[0] if len(df) > 1 else current_subscribers
            total_growth = current_subscribers - previous_subscribers
            growth_percentage = (total_growth / previous_subscribers * 100) if previous_subscribers > 0 else 0
            
            logger.info(f"Поточна кількість підписників: {current_subscribers}")
            logger.info(f"Загальний приріст: {total_growth} ({growth_percentage:.2f}%)")
            
            return {
                'current_subscribers': int(current_subscribers),
                'total_growth': int(total_growth),
                'growth_percentage': float(growth_percentage),
                'daily_growth': df[['date', 'growth']].to_dict('records'),
                'plot': buf
            }
            
        except Exception as e:
            logger.error(f"Помилка при аналізі приросту аудиторії: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    async def analyze_post_performance(self):
        """Аналіз ефективності постів"""
        logger.info("Початок аналізу ефективності постів")
        
        try:
            cursor = self.db_conn.cursor()
            cursor.execute("""
                SELECT p.message_id, p.date, p.text, p.views, p.forwards, 
                       SUM(r.count) as total_reactions
                FROM posts p
                LEFT JOIN reactions r ON p.message_id = r.message_id
                GROUP BY p.message_id
                ORDER BY p.date DESC
            """)
            
            rows = cursor.fetchall()
            
            if not rows:
                logger.info("Немає даних для аналізу ефективності постів")
                return None
            
            logger.info(f"Отримано {len(rows)} постів для аналізу ефективності")
            
            # Створення DataFrame для аналізу
            df = pd.DataFrame(rows, columns=['message_id', 'date', 'text', 'views', 'forwards', 'total_reactions'])
            df['date'] = pd.to_datetime(df['date'])
            
            # Розрахунок показників ефективності
            df['engagement_rate'] = ((df['total_reactions'].fillna(0) + df['forwards'].fillna(0)) / df['views'].fillna(1)) * 100
            df['text_length'] = df['text'].fillna('').apply(len)
            
            # Аналіз за часом публікації
            df['hour'] = df['date'].dt.hour
            hourly_performance = df.groupby('hour')['engagement_rate'].mean().sort_values(ascending=False)
            
            logger.info("Аналіз ефективності за годинами:")
            for hour, rate in hourly_performance.items()[:5]:
                logger.info(f"{hour}:00 - {rate:.2f}% залученість")
            
            # Аналіз за довжиною тексту
            df['length_category'] = pd.cut(df['text_length'], 
                                          bins=[0, 100, 500, 1000, 2000, float('inf')],
                                          labels=['Дуже короткий', 'Короткий', 'Середній', 'Довгий', 'Дуже довгий'])
            
            length_performance = df.groupby('length_category')['engagement_rate'].mean().sort_values(ascending=False)
            
            logger.info("Аналіз ефективності за довжиною тексту:")
            for length, rate in length_performance.items():
                logger.info(f"{length} - {rate:.2f}% залученість")
            
            # Кластеризація постів за показниками (машинне навчання)
            if len(df) >= 5:  # Мінімальна кількість для кластеризації
                # Вибір показників для кластеризації
                features = df[['views', 'total_reactions', 'forwards', 'text_length']].fillna(0)
                
                # Стандартизація даних
                scaler = StandardScaler()
                scaled_features = scaler.fit_transform(features)
                
                # Визначення оптимальної кількості кластерів (від 2 до 5)
                n_clusters = min(5, len(df) // 2)
                n_clusters = max(2, n_clusters)  # Мінімум 2 кластери
                
                # Кластеризація
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                df['cluster'] = kmeans.fit_predict(scaled_features)
                
                # Аналіз кластерів
                cluster_stats = df.groupby('cluster').agg({
                    'views': 'mean',
                    'total_reactions': 'mean',
                    'forwards': 'mean',
                    'engagement_rate': 'mean',
                    'text_length': 'mean',
                    'message_id': 'count'
                }).sort_values(by='engagement_rate', ascending=False)
                
                logger.info("Результати кластеризації постів:")
                for cluster, stats in cluster_stats.iterrows():
                    logger.info(f"Кластер {cluster}:")
                    logger.info(f"  Кількість постів: {stats['message_id']}")
                    logger.info(f"  Середній рівень залученості: {stats['engagement_rate']:.2f}%")
                    logger.info(f"  Середні перегляди: {stats['views']:.0f}")
                    logger.info(f"  Середня довжина тексту: {stats['text_length']:.0f} символів")
                
                # Знаходження найуспішніших постів у найкращому кластері
                best_cluster = cluster_stats.index[0]
                top_posts = df[df['cluster'] == best_cluster].sort_values(by='engagement_rate', ascending=False).head(5)
                
                logger.info("Топ-5 найуспішніших постів:")
                for i, (idx, post) in enumerate(top_posts.iterrows(), 1):
                    logger.info(f"{i}. ID: {post['message_id']}, Залученість: {post['engagement_rate']:.2f}%")
                    logger.info(f"   Перегляди: {post['views']}, Реакції: {post['total_reactions']}")
                    preview = post['text'][:100] + "..." if len(post['text']) > 100 else post['text']
                    logger.info(f"   Текст: {preview}")
            
            # Створення графіків
            plt.figure(figsize=(15, 10))
            
            # Графік залученості за годинами
            plt.subplot(2, 2, 1)
            hourly_data = df.groupby('hour')['engagement_rate'].mean()
            sns.barplot(x=hourly_data.index, y=hourly_data.values)
            plt.title('Залученість за годинами публікації')
            plt.xlabel('Година')
            plt.ylabel('Середня залученість (%)')
            
            # Графік залученості за довжиною тексту
            plt.subplot(2, 2, 2)
            sns.boxplot(x='length_category', y='engagement_rate', data=df)
            plt.title('Залученість за довжиною тексту')
            plt.xlabel('Категорія довжини')
            plt.ylabel('Залученість (%)')
            plt.xticks(rotation=45)
            
            # Графік динаміки переглядів за часом
            plt.subplot(2, 2, 3)
            # Сортуємо за датою
            temp_df = df.sort_values('date')
            sns.lineplot(x=range(len(temp_df)), y='views', data=temp_df)
            plt.title('Динаміка переглядів постів')
            plt.xlabel('Номер поста (хронологічно)')
            plt.ylabel('Кількість переглядів')
            
            # Графік співвідношення переглядів і реакцій
            plt.subplot(2, 2, 4)
            sns.scatterplot(x='views', y='total_reactions', size='forwards', data=df)
            plt.title('Співвідношення переглядів і реакцій')
            plt.xlabel('Перегляди')
            plt.ylabel('Кількість реакцій')
            
            plt.tight_layout()
            
            # Збереження графіку
            buf = BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            
            logger.info("Аналіз ефективності постів успішно завершено")
            
            return {
                'hourly_performance': hourly_performance.to_dict(),
                'length_performance': length_performance.to_dict(),
                'top_posts': top_posts[['message_id', 'date', 'views', 'total_reactions', 'engagement_rate']].to_dict('records') if len(df) >= 5 else [],
                'plot': buf
            }
            
        except Exception as e:
            logger.error(f"Помилка при аналізі ефективності постів: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    async def search_relevant_channels(self, keywords, limit=20):
        """
        Пошук релевантних каналів за ключовими словами
        
        :param keywords: Список ключових слів для пошуку
        :param limit: Максимальна кількість каналів для повернення
        """
        logger.info(f"Початок пошуку релевантних каналів за ключовими словами: {keywords}")
        
        try:
            relevant_channels = []
            
            for keyword in keywords:
                logger.info(f"Пошук каналів за ключовим словом: {keyword}")
                
                # Пошук по ключовому слову в Telegram
                search_results = await self.client.get_dialogs(limit=100)
                
                for dialog in search_results:
                    if not dialog.is_channel:
                        continue
                    
                    channel_title = dialog.title.lower()
                    
                    # Перевірка на наявність ключового слова
                    if keyword.lower() in channel_title:
                        logger.info(f"Знайдено канал: {dialog.title}")
                        
                        # Отримання повної інформації про канал
                        try:
                            channel_entity = await self.client.get_entity(dialog.id)
                            channel_full = await self.client(GetFullChannelRequest(channel=channel_entity))
                            subscribers = channel_full.full_chat.participants_count
                            
                            # Розрахунок релевантності (базова версія)
                            # Можна доопрацювати з більш складними алгоритмами
                            relevance_score = 1.0
                            
                            # Додаємо в список, якщо не дублікат
                            channel_info = {
                                'channel_id': dialog.id,
                                'channel_title': dialog.title,
                                'subscribers': subscribers,
                                'relevance_score': relevance_score
                            }
                            
                            # Перевірка на дублікати
                            if not any(ch['channel_id'] == dialog.id for ch in relevant_channels):
                                relevant_channels.append(channel_info)
                                
                                # Збереження в базу даних
                                cursor = self.db_conn.cursor()
                                cursor.execute("""
                                    INSERT OR REPLACE INTO relevant_channels 
                                    (channel_id, channel_title, subscribers, relevance_score) 
                                    VALUES (?, ?, ?, ?)
                                """, (dialog.id, dialog.title, subscribers, relevance_score))
                                self.db_conn.commit()
                        
                        except Exception as e:
                            logger.error(f"Помилка при отриманні інформації про канал {dialog.title}: {str(e)}")
                            continue
            
            # Сортування за релевантністю та обмеження кількості
            relevant_channels.sort(key=lambda x: x['relevance_score'], reverse=True)
            relevant_channels = relevant_channels[:limit]
            
            logger.info(f"Знайдено {len(relevant_channels)} релевантних каналів")
            
            return relevant_channels
            
        except Exception as e:
            logger.error(f"Помилка при пошуку релевантних каналів: {str(e)}")
            logger.error(traceback.format_exc())
            return []
    
    async def schedule_post(self, post_text, scheduled_time, media_path=None, buttons=None):
        """
        Планування нового поста
        
        :param post_text: Текст повідомлення
        :param scheduled_time: Час публікації (у форматі 'YYYY-MM-DD HH:MM:SS')
        :param media_path: Шлях до медіа-файлу (опційно)
        :param buttons: Список кнопок у форматі JSON (опційно)
        """
        logger.info(f"Планування нового поста на {scheduled_time}")
        
        try:
            # Перетворення кнопок у JSON, якщо вони надані
            buttons_json = json.dumps(buttons) if buttons else None
            
            # Збереження запланованого поста в базу даних
            cursor = self.db_conn.cursor()
            cursor.execute("""
                INSERT INTO scheduled_posts (scheduled_time, post_text, media_path, buttons, status)
                VALUES (?, ?, ?, ?, 'pending')
            """, (scheduled_time, post_text, media_path, buttons_json))
            
            post_id = cursor.lastrowid
            self.db_conn.commit()
            
            logger.info(f"Пост успішно запланований, ID: {post_id}")
            
            return post_id
            
        except Exception as e:
            logger.error(f"Помилка при плануванні поста: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    async def process_scheduled_posts(self):
        """Обробка та публікація запланованих постів"""
        logger.info("Перевірка запланованих постів")
        
        try:
            # Отримання поточного часу
            now = datetime.datetime.now()
            
            # Отримання запланованих постів, час яких вже настав
            cursor = self.db_conn.cursor()
            cursor.execute("""
                SELECT id, scheduled_time, post_text, media_path, buttons
                FROM scheduled_posts
                WHERE status = 'pending' AND datetime(scheduled_time) <= datetime(?)
            """, (now.strftime("%Y-%m-%d %H:%M:%S"),))
            
            posts = cursor.fetchall()
            
            if not posts:
                logger.info("Немає постів для публікації")
                return
            
            logger.info(f"Знайдено {len(posts)} постів для публікації")
            
            for post in posts:
                post_id, scheduled_time, post_text, media_path, buttons_json = post
                
                logger.info(f"Публікація поста ID: {post_id}")
                
                try:
                    # Підготовка кнопок, якщо вони є
                    buttons = None
                    if buttons_json:
                        try:
                            buttons_data = json.loads(buttons_json)
                            # Тут можна додати код для створення власних кнопок Telegram
                        except json.JSONDecodeError:
                            logger.error(f"Помилка декодування JSON для кнопок поста {post_id}")
                    
                    # Публікація поста
                    if media_path and os.path.exists(media_path):
                        # Відправка з медіа
                        await self.client.send_file(
                            self.channel,
                            media_path,
                            caption=post_text,
                            buttons=buttons
                        )
                        logger.info(f"Опубліковано пост з медіа: {media_path}")
                    else:
                        # Відправка тільки тексту
                        await self.client.send_message(
                            self.channel,
                            post_text,
                            buttons=buttons
                        )
                        logger.info(f"Опубліковано текстовий пост")
                    
                    # Оновлення статусу поста
                    cursor.execute(
                        "UPDATE scheduled_posts SET status = 'published' WHERE id = ?",
                        (post_id,)
                    )
                    self.db_conn.commit()
                    
                    logger.info(f"Пост успішно опубліковано, ID: {post_id}")
                    
                except Exception as e:
                    logger.error(f"Помилка при публікації поста {post_id}: {str(e)}")
                    logger.error(traceback.format_exc())
                    
                    # Встановлення статусу помилки
                    cursor.execute(
                        "UPDATE scheduled_posts SET status = 'error' WHERE id = ?",
                        (post_id,)
                    )
                    self.db_conn.commit()
            
        except Exception as e:
            logger.error(f"Помилка при обробці запланованих постів: {str(e)}")
            logger.error(traceback.format_exc())
    
    async def predict_optimal_posting_time(self):
        """Прогнозування оптимального часу для публікації постів на основі статистики"""
        logger.info("Початок прогнозування оптимального часу для публікацій")
        
        try:
            cursor = self.db_conn.cursor()
            cursor.execute("""
                SELECT p.message_id, p.date, p.views, p.forwards, 
                       SUM(CASE WHEN r.count IS NULL THEN 0 ELSE r.count END) as total_reactions
                FROM posts p
                LEFT JOIN reactions r ON p.message_id = r.message_id
                GROUP BY p.message_id
                ORDER BY p.date
            """)
            
            rows = cursor.fetchall()
            
            if not rows or len(rows) < 10:  # Потрібно достатньо даних для аналізу
                logger.info("Недостатньо даних для прогнозування оптимального часу публікацій")
                return None
            
            logger.info(f"Отримано {len(rows)} постів для аналізу оптимального часу")
            
            # Створення DataFrame для аналізу
            df = pd.DataFrame(rows, columns=['message_id', 'date', 'views', 'forwards', 'total_reactions'])
            df['date'] = pd.to_datetime(df['date'])
            
            # Додавання часових ознак
            df['hour'] = df['date'].dt.hour
            df['day_of_week'] = df['date'].dt.dayofweek  # 0 = понеділок, 6 = неділя
            
            # Розрахунок метрики залученості
            df['engagement'] = df['total_reactions'] + df['forwards']
            df['engagement_rate'] = (df['engagement'] / df['views'] * 100).fillna(0)
            
            # Аналіз за годинами
            hourly_performance = df.groupby('hour')['engagement_rate'].mean()
            top_hours = hourly_performance.sort_values(ascending=False).head(3)
            
            # Аналіз за днями тижня
            daily_performance = df.groupby('day_of_week')['engagement_rate'].mean()
            top_days = daily_performance.sort_values(ascending=False).head(3)
            
            # Перетворення номерів днів у назви
            day_names = {
                0: 'Понеділок',
                1: 'Вівторок',
                2: 'Середа',
                3: 'Четвер',
                4: 'П\'ятниця',
                5: 'Субота',
                6: 'Неділя'
            }
            
            # Найкращі комбінації день-година
            combined_performance = df.groupby(['day_of_week', 'hour'])['engagement_rate'].mean().reset_index()
            top_combinations = combined_performance.sort_values('engagement_rate', ascending=False).head(5)
            
            # Створення рекомендацій
            recommendations = []
            
            logger.info("Найкращі години для публікацій:")
            for hour, rate in top_hours.items():
                logger.info(f"{hour}:00 - Середня залученість: {rate:.2f}%")
                recommendations.append({
                    'hour': int(hour),
                    'day': None,
                    'engagement_rate': float(rate),
                    'recommendation': f"Публікуйте о {hour}:00 для кращої залученості"
                })
            
            logger.info("Найкращі дні тижня для публікацій:")
            for day, rate in top_days.items():
                day_name = day_names.get(day, f"День {day}")
                logger.info(f"{day_name} - Середня залученість: {rate:.2f}%")
                recommendations.append({
                    'hour': None,
                    'day': int(day),
                    'day_name': day_name,
                    'engagement_rate': float(rate),
                    'recommendation': f"Публікуйте в {day_name} для кращої залученості"
                })
            
            logger.info("Найкращі комбінації день-година для публікацій:")
            for _, row in top_combinations.iterrows():
                day_name = day_names.get(row['day_of_week'], f"День {row['day_of_week']}")
                logger.info(f"{day_name}, {row['hour']}:00 - Середня залученість: {row['engagement_rate']:.2f}%")
                recommendations.append({
                    'hour': int(row['hour']),
                    'day': int(row['day_of_week']),
                    'day_name': day_name,
                    'engagement_rate': float(row['engagement_rate']),
                    'recommendation': f"Оптимальний час: {day_name}, {row['hour']}:00"
                })
            
            # Створення графіків
            plt.figure(figsize=(15, 10))
            
            # Графік ефективності за годинами
            plt.subplot(2, 2, 1)
            hours = list(range(24))
            performance = [hourly_performance.get(hour, 0) for hour in hours]
            sns.barplot(x=hours, y=performance)
            plt.title('Ефективність публікацій за годинами')
            plt.xlabel('Година')
            plt.ylabel('Середня залученість (%)')
            
            # Графік ефективності за днями тижня
            plt.subplot(2, 2, 2)
            days = list(range(7))
            day_labels = [day_names[day] for day in days]
            performance = [daily_performance.get(day, 0) for day in days]
            sns.barplot(x=days, y=performance)
            plt.xticks(days, day_labels, rotation=45)
            plt.title('Ефективність публікацій за днями тижня')
            plt.xlabel('День тижня')
            plt.ylabel('Середня залученість (%)')
            
            # Теплова карта ефективності день-година
            plt.subplot(2, 1, 2)
            pivot_data = combined_performance.pivot(index='day_of_week', columns='hour', values='engagement_rate')
            sns.heatmap(pivot_data, cmap='YlGnBu', annot=True, fmt='.1f')
            plt.title('Теплова карта ефективності (день тижня vs година)')
            plt.xlabel('Година')
            plt.ylabel('День тижня')
            plt.yticks(range(7), [day_names[i] for i in range(7)])
            
            plt.tight_layout()
            
            # Збереження графіку
            buf = BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            
            logger.info("Прогнозування оптимального часу публікацій успішно завершено")
            
            return {
                'top_hours': top_hours.to_dict(),
                'top_days': {day_names[day]: rate for day, rate in top_days.to_dict().items()},
                'top_combinations': top_combinations.to_dict('records'),
                'recommendations': recommendations,
                'plot': buf
            }
            
        except Exception as e:
            logger.error(f"Помилка при прогнозуванні оптимального часу: {str(e)}")
            logger.error(traceback.format_exc())
            return None

    def close(self):
        """Завершення роботи сервера"""
        logger.info("Завершення роботи серверної частини")
        
        if self.db_conn:
            self.db_conn.close()
            logger.info("З'єднання з базою даних закрито")
        
        if self.client:
            self.client.disconnect()
            logger.info("З'єднання з Telegram API закрито")


async def run_server():
    """Запуск серверної частини"""
    logger.info("Запуск серверної частини чат-бота")
    
    # Отримання ID каналу від користувача
    channel_id = CHANNEL_ID or input("Введіть ID або юзернейм каналу (наприклад, @mychannel або -1001234567890): ")
    
    # Видалення символу @ з юзернейму каналу, якщо він є
    if channel_id.startswith('@'):
        channel_id = channel_id[1:]
    
    # Ініціалізація сервера
    server = TelegramServer(API_ID, API_HASH, SESSION_NAME, channel_id)
    
    # Підключення до Telegram API
    connected = await server.connect()
    
    if not connected:
        logger.error("Не вдалося з'єднатися з Telegram API. Завершення роботи.")
        server.close()
        return
    
    try:
        # Основний цикл роботи сервера
        while True:
            try:
                # Збір статистики каналу
                await server.collect_channel_stats()
                
                # Обробка запланованих постів
                await server.process_scheduled_posts()
                
                # Аналіз даних (раз на день)
                current_time = datetime.datetime.now()
                if current_time.hour == 3 and current_time.minute < 15:  # Аналіз о 3:00-3:15
                    logger.info("Запуск щоденного аналізу")
                    await server.analyze_audience_growth()
                    await server.analyze_emoji_reactions()
                    await server.analyze_post_performance()
                    await server.predict_optimal_posting_time()
                
                # Затримка перед наступною ітерацією
                await asyncio.sleep(900)  # 15 хвилин
                
            except Exception as e:
                logger.error(f"Помилка в основному циклі: {str(e)}")
                logger.error(traceback.format_exc())
                await asyncio.sleep(60)  # Очікування 1 хвилину перед повторною спробою
    
    except KeyboardInterrupt:
        logger.info("Отримано сигнал переривання. Завершення роботи.")
    
    finally:
        # Завершення роботи
        server.close()
        logger.info("Сервер зупинено")


if __name__ == "__main__":
    try:
        # Перевірка наявності необхідних бібліотек
        required_libraries = ['telethon', 'pandas', 'matplotlib', 'seaborn', 'sklearn']
        missing_libraries = []
        
        for lib in required_libraries:
            try:
                __import__(lib)
                print(f"Successfully imported {lib}")
            except ImportError as e:
                missing_libraries.append(lib)
                print(f"Failed to import {lib}: {e}")
        
        if missing_libraries:
            logger.error(f"Відсутні необхідні бібліотеки: {', '.join(missing_libraries)}")
            logger.error("Встановіть їх за допомогою pip:")
            logger.error(f"pip install {' '.join(missing_libraries)}")
            sys.exit(1)
        
        # Відображення інформації про конфігурацію
        logger.info("========================")
        logger.info("Запуск бота з наступними параметрами:")
        logger.info(f"API_ID: {API_ID}")
        logger.info(f"API_HASH: {API_HASH}")
        logger.info(f"SESSION_NAME: {SESSION_NAME}")
        logger.info("========================")
        
        # Запуск серверної частини
        asyncio.run(run_server())
    except Exception as e:
        logger.critical(f"Критична помилка: {str(e)}")
        logger.critical(traceback.format_exc())