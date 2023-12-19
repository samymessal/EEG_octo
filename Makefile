all: 
	docker build -t eeg_noteboook .
	docker run -p 8888:8888 -v ./:/app --name eeg_noteboook eeg_noteboook

build:
	docker build -t eeg_noteboook .
	
run:
	docker run -p 8888:8888 -v ./:/app --name eeg_noteboook eeg_noteboook

stop:
	docker container stop eeg_noteboook

exec:
	docker exec -it eeg_noteboook bash
	
logs:
	docker logs eeg_noteboook

clean:
	docker container stop eeg_noteboook

fclean:
	@docker system prune -af

re: fclean all

.Phony: all logs clean fclean