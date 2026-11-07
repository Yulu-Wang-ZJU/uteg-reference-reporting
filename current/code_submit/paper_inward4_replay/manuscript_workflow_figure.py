"""Preserve the editable workflow layout and correct input-version labels."""
from zipfile import ZipFile,ZIP_DEFLATED
from lxml import etree
from . import core as c,current_suite as suite

CAPTION_NOTES = [
    'MA: B-only RBF; other readouts: Ridge',
    'Measured signals at 5, 10 and 15 min',
    'Illustrative signals',
    'Update the same record at each checkpoint',
    'Risk models: cross-fitted errors',
    'Models and gates fitted on training records',
    'Main comparisons: Huzhou → SAHZU and Huzhou + Hubei → SAHZU; target-only comparator, same SAHZU test cohort',
]

def refine_labels(root):
    ns={'p':'http://schemas.openxmlformats.org/presentationml/2006/main',
        'a':'http://schemas.openxmlformats.org/drawingml/2006/main'}
    for node in root.findall('.//a:t',ns):
        if node.text=='Within range':node.text='Normal'
    def shape(sid):
        return root.xpath(f'.//p:cNvPr[@id="{sid}"]',namespaces=ns)[0].getparent().getparent()
    # Separate s from the dashed threshold without changing its position/value.
    shape(192).find('p:spPr/a:xfrm/a:off',ns).set('x',str(9744075-201168))
    # Allow room for the subscript in the readout box and historical-pair cells.
    for sid,y,size in [(328,6754500,1250),(322,9597771,1150),(323,9597771,1150)]:
        node=shape(sid)
        node.find('p:spPr/a:xfrm/a:off',ns).set('y',str(y))
        for prop in node.xpath('.//a:rPr|.//a:defRPr|.//a:endParaRPr',namespaces=ns):
            prop.set('sz',str(size))

def run():
    src=c.WS/'outputs/uteg_nc_goal_20261030/manuscript/figures/newB_20261103/Figure1_workflow_ma.pptx'
    out=suite.ROOT/'manuscript_sync/deliverables/figures/newB_20261103/Figure1_workflow_ma.pptx'
    changes={'Instrument-native output, 1 s/sample':'Measured U-TEG signal, 1 s/sample',
        'Same record, observed prefix only':'Measured signals at 5, 10 and 15 min',
        'Frozen mapping model: source or target-only':'MA: B-only RBF; other readouts: Ridge',
        'Historical learning only':'Models and gates fitted on training records',
        'Update at each checkpoint':'Current fixed B version',
        'Historical target adaptation':'Target-centre correction',
        '03  Reliability / 04  Reporting':'03  Reporting Agent',
        'Reliability score':'Risk assessment',
        'Reliability learning: cross-fitted errors':'Risk models: cross-fitted errors',
        'Historical paired U-TEG + TEG':'Historical U-TEG and TEG records',
        'Historical target pairs':'Historical target records'}
    changed=[];removed=[]
    with ZipFile(src) as inp,ZipFile(out,'w',ZIP_DEFLATED) as dest:
        for name in inp.namelist():
            data=inp.read(name)
            if name=='ppt/slides/slide1.xml':
                root=etree.fromstring(data)
                for node in root.findall('.//{http://schemas.openxmlformats.org/drawingml/2006/main}t'):
                    if node.text in changes:changed.append(node.text);node.text=changes[node.text]
                    if node.text in CAPTION_NOTES:removed.append(node.text);node.text=''
                if changed:refine_labels(root)
                data=etree.tostring(root,encoding='UTF-8',xml_declaration=True,standalone=True)
            dest.writestr(name,data)
    # The retained current master already contains these labels. Preserve its
    # layout on subsequent exports instead of requiring the obsolete master.
    assert not changed or set(changed)==set(changes)
    c.write(suite.ROOT/'manuscript_sync/FIGURE1_EDITS.json',dict(text_changes=changes,
        caption_notes_removed=CAPTION_NOTES,notes_removed_in_this_run=removed,
        state_label='Normal',agent_roles=['Measurement','Translation','Reporting'],
        reporting_scope='Risk assessment and report/observe/withhold decisions within the reporting agent; unchanged computation',
        pairing_scope='Measurements linked by blood sample, not pointwise time alignment',
        layout='s moved left of the unchanged threshold; alpha_R labels fitted inside their boxes'))

if __name__=='__main__':run()
