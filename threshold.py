import pickle as p

if __name__ == '__main__':
    #L1 Tamb Tsuc Tliq Psuc Tsh Pliq
    import argparse
    parser = argparse.ArgumentParser(description="Script para gerar arquivo de threshold para máquina indicada.")
    parser.add_argument('-d',action='append', dest='dev_id', type=str, help='DEV_ID da máquina.')
    args = parser.parse_args()
    #print(args)

    dev_id = args.dev_id[0]
    threshold = dict()

    threshold['L1'] = float(input("L1:\n"))
    threshold['Tamb'] = float(input("Tamb:\n"))
    threshold['Tsuc'] = float(input("Tsuc:\n"))
    threshold['Tliq'] = float(input("Tliq:\n"))
    threshold['Psuc'] = float(input("Psuc:\n"))
    threshold['Tsh'] = float(input("Tsh:\n"))
    threshold['Pliq'] = float(input("Pliq:\n"))

    f = open(f'models/{dev_id}/{dev_id}_threshold.p', "wb")
    p.dump(threshold, f)



