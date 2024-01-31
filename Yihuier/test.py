
if __name__ == "__main__":

    money = 4
    count = money//2
    bottle = count
    cap = count

    while cap//4 >0 or bottle//2 >0:
        cap_num = cap//4
        bottle_num = bottle//2
        count_num = cap_num + bottle_num
        cap = cap%4 + count_num
        bottle = bottle%2 + count_num
        count = count+count_num
        print(count,cap,bottle,cap_num,bottle_num,count_num)

    # while cap//3 >0 or bottle//1 >0:
    #     cap_num = cap//3
    #     bottle_num = bottle//1
    #     count_num = cap_num + bottle_num
    #     cap = cap%3 + count_num
    #     bottle = bottle%1 + count_num
    #     count = count+count_num
    #     print(count,cap,bottle,cap_num,bottle_num,count_num)
