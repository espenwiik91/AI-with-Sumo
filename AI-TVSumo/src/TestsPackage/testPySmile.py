# License issued by BayesFusion Licensing Server
# The code below must be executed before any PySMILE object is created.
# You can use "import pysmile_license" or copy the pysmile.License
# call into your Python source code.
import pysmile
import pysmile_license  # @UnusedImport

def main():

    print("vi kjoerer")
    feverNet = pysmile.Network()
    print("lagde nettverk")
    feverNet.read_file("FluModel.xdsl")
    feverNet.set_evidence("flu", "True")
    feverNet.update_beliefs()
    beliefs = feverNet.get_node_value("HighTemp")
    for i in range(0, len(beliefs)):
        print(feverNet.get_outcome_id("HighTemp",i) + "=" + str(beliefs[i]))


if __name__ == '__main__':
    main()
