from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget


class SummaryTable(QWidget):
    def __init__(self):
        super().__init__()

        self.table = QTableWidget()
        self.table.setColumnCount(20)

        self.add_headers()

        """
        self.add_data([
            "VpToxSp_GC_240729_29/1", "2.5", "29.37", "0.31", "0.939", "99.9", "8.3",
            "2.5", "2.57", "0.834", "0.823", "99.1", "8.7",
            "P 2 2 2", "69.4972", "92.6067", "95.352", "90.0", "90.0", "90.0"
        ])
        """

        layout = QVBoxLayout()
        layout.addWidget(self.table)
        self.setLayout(layout)

    def add_headers(self):
        self.table.insertRow(self.table.rowCount())  # Add a new row
        self.table.setItem(0, 0, QTableWidgetItem("#"))
        self.table.setItem(0, 1, QTableWidgetItem("Overall"))
        self.table.setSpan(0, 1, 1, 6)  # Colspan of 6 for "Overall"
        self.table.setItem(0, 7, QTableWidgetItem("Outer Shell"))
        self.table.setSpan(0, 7, 1, 6)  # Colspan of 6 for "Outer Shell"
        self.table.setItem(0, 13, QTableWidgetItem(""))  # Empty cell for rest

        self.table.insertRow(self.table.rowCount())
        headers = [
            "Sample Path", "Hi", "Lo", "R_mrg", "cc12", "comp", "mult",
            "Hi", "Lo", "R_mrg", "cc12", "comp", "mult",
            "symm", "a", "b", "c", "alpha", "beta", "gamma"
        ]
        for i, header in enumerate(headers):
            self.table.setItem(1, i, QTableWidgetItem(header))

        self.table.resizeColumnsToContents()

    def add_data(self, row_data):
        row_position = self.table.rowCount()
        self.table.insertRow(row_position)

        for i, value in enumerate(row_data):
            self.table.setItem(row_position, i, QTableWidgetItem(str(value)))

        self.table.resizeColumnsToContents()

    def clear_data(self):
        self.table.clear()
        self.table.setRowCount(0) 
        self.add_headers()
        
        
class CollectionTable(QWidget):
    def __init__(self):
        super().__init__()

        self.table = QTableWidget()
        self.table.setColumnCount(7)  
        self.table.setRowCount(1)

        self.add_headers()

        #self.add_data([
        #    "1454a4cc-4e60-47c5-8c78-bc3f63d5d030", "2024-08-01 13:04:36", "0.0", "225.0", "0.1", "0.005", "225.0"
        #])

        # Set up the layout
        layout = QVBoxLayout()
        layout.addWidget(self.table)
        self.setLayout(layout)

    def add_headers(self):
        headers = [
            "Collection ID", "Time", "Sweep start", "Sweep end",
            "Image width", "Exposure time", "Detector distance"
        ]

        for i, header in enumerate(headers):
            header_item = QTableWidgetItem(header)
            #header_item.setTextAlignment(Qt.AlignCenter)  # Center align text
            self.table.setHorizontalHeaderItem(i, header_item)

        self.table.resizeColumnsToContents()

    def add_data(self, row_data):
        # Add data to the table row
        for i, value in enumerate(row_data):
            self.table.setItem(0, i, QTableWidgetItem(str(value)))  # Insert data into the first (and only) row

        # Optionally resize the columns to fit the data better
        self.table.resizeColumnsToContents()

    def clear_data(self):
        self.table.clearContents()
